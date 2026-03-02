import os

# Disable Rust-level tokenizer parallelism to avoid deadlocks when the process
# is forked by DataLoader workers or DDP (the tokenizer is used inside collate_fn).
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from utils.create_datasets import create_dataset, filter_qrels_by_length

from functools import partial
import numpy as np
import torch.distributed as dist

from datasets import Dataset, Features, Value, Sequence
from tasks.load_datasets import load_task_data
from tasks import get_task
from tasks.task_categories import get_category_path
from utils.dataloader_helpers import LenghtSortedSampler, collate_fn_with_padding
from pathlib import Path
import time
from inference.helpers import encode, search
from dataclasses import dataclass
import json
from utils.helpers import print_memory_consumed, return_formatted
import pyarrow.compute as pc

# from .helpers import last_token_pool as _default_pool


# Maximum number of queries to process in a single mine-one pass.
# When the total number of unique queries exceeds this, the full
# encode → search → get_hard_negatives → save pipeline is run
# iteratively in chunks of this size.  This avoids the need for
# stream_to_cpu which may not be available on every server.
# ITERATIVE_ENCODE_THRESHOLD = 10**6


def _get_hidden_size(model):
    """Unwrap DDP and EncoderWithPooling to reach the model with config."""
    m = model.module if hasattr(model, "module") else model
    while hasattr(m, "model") and not hasattr(m, "config"):
        m = m.model
    return m.config.hidden_size


def estimate_chunk_sizes(query_embeddings, max_corpus_chunk=5 * 10**4):
    """Estimate corpus and query chunk sizes to stay within GPU memory.

    When the full query set fits comfortably, *query_chunk_size* equals
    *n_queries* (i.e. no query chunking).  For very large query sets the
    function picks a smaller *query_chunk_size* so that the similarity
    matrix ``[query_chunk, corpus_chunk]`` stays within the memory budget.

    Handles both GPU-resident and CPU-resident query embeddings.

    Returns:
        (corpus_chunk_size, query_chunk_size)
    """
    free_mem, _ = torch.cuda.mem_get_info()
    n_queries = query_embeddings.shape[0]
    elem_size = query_embeddings.element_size()
    dim = query_embeddings.shape[1]

    bytes_per_doc = dim * elem_size  # one corpus embedding vector
    budget = int(0.8 * free_mem)
    queries_on_cpu = not query_embeddings.is_cuda

    if queries_on_cpu:
        # Query embeddings are NOT on GPU, so GPU budget is only for:
        #   corpus_chunk embeddings  +  query_slice on GPU  +  score matrix
        corpus_chunk = min(max_corpus_chunk, 5 * 10**4)
        corpus_bytes = corpus_chunk * bytes_per_doc
        remaining = budget - corpus_bytes
        # Each query row costs: dim * elem_size (embedding slice) + corpus_chunk * elem_size (score row)
        bytes_per_query_row = (dim + corpus_chunk) * elem_size
        query_chunk = remaining // max(1, bytes_per_query_row)
        query_chunk = max(10**4, min(int(query_chunk), n_queries))
        return corpus_chunk, query_chunk

    # --- queries on GPU ---------------------------------------------------
    bytes_per_sim_col = n_queries * elem_size  # one column of the full sim matrix

    # try without query chunking first
    corpus_chunk = budget // max(1, bytes_per_doc + bytes_per_sim_col)
    corpus_chunk = max(1000, min(int(corpus_chunk), max_corpus_chunk))

    total_needed = n_queries * corpus_chunk * elem_size + corpus_chunk * bytes_per_doc
    if total_needed <= budget:
        return corpus_chunk, n_queries

    # need query chunking
    corpus_chunk = min(max_corpus_chunk, 10**4)
    remaining = budget - corpus_chunk * bytes_per_doc
    query_chunk = remaining // max(1, corpus_chunk * elem_size)
    query_chunk = max(10**4, min(int(query_chunk), n_queries))

    return corpus_chunk, query_chunk


@dataclass
class TripletStats:
    less_than_24: int = 0
    less_than_15: int = 0
    less_than_7: int = 0
    empty_entries: int = 0
    total_queries: int = 0

    def update(self, num_negatives: int, q_id_counts):
        """Update stats based on number of negatives"""
        if num_negatives < 24:
            self.less_than_24 += q_id_counts
        if num_negatives < 15:
            self.less_than_15 += q_id_counts
        if num_negatives < 7:
            self.less_than_7 += q_id_counts
        if num_negatives == 0:
            self.empty_entries += q_id_counts
            # if self.rank == 0:
            # print("Found empty entry", q_id_counts)

    def merge(self, other: "TripletStats"):
        """Accumulate stats from another TripletStats instance."""
        self.less_than_24 += other.less_than_24
        self.less_than_15 += other.less_than_15
        self.less_than_7 += other.less_than_7
        self.empty_entries += other.empty_entries
        self.total_queries += other.total_queries

    def to_dict(self):
        """Return a JSON-serialisable dict of these stats."""
        return {
            "num_triples": self.total_queries,
            "num_empty_negative_entries": self.empty_entries,
            "num_with_7_hard_negatives": self.total_queries - self.less_than_7,
            "num_with_15_hard_negatives": self.total_queries - self.less_than_15,
            "num_with_24_hard_negatives": self.total_queries - self.less_than_24,
        }


def update_dataset_dict(
    dataset_dict,
    qrels,
    negatives,
    has_title,
    corpus_dict,
    query_dict,
    subtask=None,
):
    """Progressively extend a dict-of-lists with data from one subtask.

    Args:
        dataset_dict: Dict of lists to update in-place. Pass ``{}`` on the first call.
        qrels: Dataset / dict with ``query_id`` and ``positive_id`` columns.
        negatives: Hard-negatives dict keyed by ``(query_id, positive_id)``.
        has_title: Whether the corpus contains titles.
        corpus_dict: Corpus mapping ``id -> {"text": ..., "title": ...}``.
        query_dict: Query mapping ``id -> {"text": ...}``.
        subtask: If not ``None``, a ``"subset"`` column is added with this value.
    """
    query_ids = qrels["query_id"]
    positive_ids = qrels["positive_id"]

    # Initialise keys on first call
    if not dataset_dict:
        dataset_dict["query_text"] = []
        dataset_dict["query_id"] = []
        dataset_dict["positive_text"] = []
        dataset_dict["positive_id"] = []
        dataset_dict["negative_text"] = []
        dataset_dict["negative_id"] = []
        if has_title:
            dataset_dict["positive_title"] = []
            dataset_dict["negative_title"] = []
        if subtask is not None:
            dataset_dict["subset"] = []

    # Skip entries with no hard negatives — they are useless for
    # contrastive training and waste disk space / memory.
    for q_id, p_id in zip(query_ids, positive_ids):
        neg = negatives.get((q_id, p_id))
        if neg is None:
            continue
        dataset_dict["query_text"].append(query_dict[q_id]["text"])
        dataset_dict["query_id"].append(q_id)
        dataset_dict["positive_text"].append(corpus_dict[p_id]["text"])
        dataset_dict["positive_id"].append(p_id)
        dataset_dict["negative_text"].append(neg["text"])
        dataset_dict["negative_id"].append(neg["id"])
        if has_title:
            dataset_dict["positive_title"].append(corpus_dict[p_id]["title"])
            dataset_dict["negative_title"].append(neg["title"])
        if subtask is not None:
            dataset_dict["subset"].append(subtask)


def _get_features_dict(has_title, has_subset):
    """Return a Features dict for the hard-negatives schema."""
    features_dict = {
        "query_text": Value("string"),
        "query_id": Value("string"),
        "positive_text": Value("string"),
        "positive_id": Value("string"),
        "negative_text": Sequence(Value("string")),
        "negative_id": Sequence(Value("string")),
    }
    if has_title:
        features_dict["positive_title"] = Value("string")
        features_dict["negative_title"] = Sequence(Value("string"))
    if has_subset:
        features_dict["subset"] = Value("string")
    return features_dict


def save_dataset_shard_to_parquet(
    dataset_dict,
    save_dir,
    shard_name,
    has_title,
    rank,
):
    """Save *dataset_dict* as a single compressed Parquet shard.

    Only rank-0 performs the I/O.  The shard is written to
    ``<save_dir>/<shard_name>.parquet`` using zstd compression,
    which typically reduces string-heavy data to ~10-20 % of the
    uncompressed Arrow size.

    The ``Dataset`` object is deleted immediately after writing so
    that peak memory stays proportional to one subtask, not the
    full accumulated dataset.
    """
    if rank != 0:
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    has_subset = "subset" in dataset_dict
    features = Features(_get_features_dict(has_title, has_subset))
    dataset = Dataset.from_dict(dataset_dict, features=features)

    parquet_path = save_path / f"{shard_name}.parquet"
    dataset.to_parquet(str(parquet_path))
    del dataset  # free Arrow memory immediately


def save_dataset_metadata(
    save_dir,
    stats,
    model_name,
    rank,
    per_subset_stats=None,
):
    """Write a JSON metadata file summarising the mined hard-negatives.

    Only rank-0 performs the I/O.
    """
    if rank != 0:
        return

    metadata = {
        "num_triples": stats.total_queries,
        "num_empty_negative_entries": stats.empty_entries,
        "num_with_7_hard_negatives": stats.total_queries - stats.less_than_7,
        "num_with_15_hard_negatives": stats.total_queries - stats.less_than_15,
        "num_with_24_hard_negatives": stats.total_queries - stats.less_than_24,
        "embedder": model_name,
    }
    if per_subset_stats:
        metadata["per_subset"] = {
            name: s.to_dict() for name, s in per_subset_stats.items()
        }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------- backward-compatible wrapper (non-subtask datasets) ----------
def save_dataset_dict_to_disk(
    dataset_dict,
    save_path,
    stats,
    model_name,
    has_title,
    rank,
    per_subset_stats=None,
):
    """Convert *dataset_dict* to a compressed Parquet file and save to *save_path*.

    Only rank-0 performs the I/O.  The ``Dataset`` object is deleted
    immediately after saving to keep peak memory low.
    """
    save_dataset_shard_to_parquet(
        dataset_dict=dataset_dict,
        save_dir=save_path,
        shard_name="data",
        has_title=has_title,
        rank=rank,
    )
    save_dataset_metadata(
        save_dir=save_path,
        stats=stats,
        model_name=model_name,
        rank=rank,
        per_subset_stats=per_subset_stats,
    )


class HardNegativesMiner:

    def __init__(
        self,
        path,
        model_name,
        tokenizer,
        task_names,
        instruction_template,
        padding_side="right",
        max_length=512,
        add_special_tokens=False,
        eot_id=None,
        iterative_encode_threshold=10**7,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = task_names
        self.padding_side = padding_side
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.eot_id = eot_id

        # Maximum number of queries to process in a single mine-one pass.
        # When the total number of unique queries exceeds this, the full
        # encode → search → get_hard_negatives → save pipeline is run
        # iteratively in chunks of this size.  This avoids the need for
        # stream_to_cpu which may not be available on every server.
        self.iterative_encode_threshold = iterative_encode_threshold

        if self.rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.instruction_template = instruction_template
        self.path = path
        self.model_name = model_name
        dist.barrier()

    def prepare_dataset(
        self,
        data_split,
        corpus_dict,
        task_metadata,
        n_positives,
    ):

        dist.barrier()
        # qrels contains query_id and positive_id pairs
        if self.rank == 0:
            print(
                f"\ntokenizing dataset: num total qrels pairs (with repetitions), {return_formatted(len(data_split['qrels']))}"
            )

        unique_queries_dataset = create_dataset(
            dataset=data_split["unique_queries"],
            task_metadata=task_metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
            max_length=self.max_length,
        )

        if self.rank == 0:
            print(
                f"num unique queries: {return_formatted(len(unique_queries_dataset))}"
            )
            print(
                f"num unique positives (first {return_formatted(n_positives)} docs in corpus)"
            )

        if self.rank == 0:
            if len(unique_queries_dataset.removed_long) > 0:
                print(
                    f"removed {len(unique_queries_dataset.removed_long)} queries exceeding max_length"
                )
            if len(unique_queries_dataset.removed_empty) > 0:
                print(
                    f"removed {len(unique_queries_dataset.removed_empty)} empty queries"
                )

        dist.barrier()
        if self.rank == 0:
            print(
                "tokenizing dataset num docs",
                return_formatted(len(data_split["corpus"])),
            )

        positive_ids = set(data_split["corpus"]["id"][:n_positives])
        assert len(positive_ids) == n_positives

        corpus_dataset = create_dataset(
            dataset=data_split["corpus"],
            task_metadata=task_metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.document,
            max_length=self.max_length,
        )
        if len(corpus_dataset.removed_ids) > 0:
            positives_to_remove = positive_ids.intersection(
                set(corpus_dataset.removed_ids)
            )
            n_positives_to_remove = len(positives_to_remove)

            if n_positives_to_remove > 0:
                n_positives -= n_positives_to_remove
                if self.rank == 0:
                    print(f"removed {n_positives_to_remove} positives")

        if self.rank == 0:
            if len(corpus_dataset.removed_long) > 0:
                print(
                    f"removed {len(corpus_dataset.removed_long)} documents exceeding max_length"
                )
            if len(corpus_dataset.removed_empty) > 0:
                print(f"removed {len(corpus_dataset.removed_empty)} empty documents")

        # Remove qrels pairs where either query or positive was removed
        # Need to check positives against corpus (first n_positives documents)
        dist.barrier()
        filtered_qrels = data_split["qrels"]
        if (
            len(unique_queries_dataset.removed_ids) > 0
            or len(corpus_dataset.removed_ids) > 0
        ):

            if self.rank == 0:
                start = time.time()
                print("removing long sequences from full queries and corpus")

            # --- Old approach (kept for reference) ---
            # t1 = time.time()
            # filtered_qrels = filter_qrels_by_length(
            #     unique_queries_dataset.removed_ids,
            #     corpus_dataset.removed_ids,
            #     data_split["qrels"],
            # )
            # dist.barrier()
            # t2 = time.time()
            # corpus_ids_set = set(corpus_dataset["id"])
            # unique_queries_dataset_set = set(unique_queries_dataset["id"])
            # dist.barrier()
            # t3 = time.time()
            # assert set(filtered_qrels["query_id"]).issubset(unique_queries_dataset_set), "filtered qrels contain query IDs not in unique queries"
            # assert set(filtered_qrels["positive_id"]).issubset(
            #     corpus_ids_set
            # ), "filtered qrels contain positive IDs not in corpus"
            # dist.barrier()
            # t4 = time.time()
            #
            # if self.rank == 0:
            #     print(
            #         f"full queries and corpus filtered in {(time.time()-start)/60:.2f}min"
            #     )
            #     print("filtering times:")
            #     print((t2-t1)/60)
            #     print((t3-t2)/60)
            #     print((t4-t3)/60)
            #     num_queries_lost = len(set(unique_queries_dataset["id"])) - len(
            #         set(filtered_qrels["query_id"])
            #     )
            #     if num_queries_lost > 0:
            #         print(
            #             f"Note: {num_queries_lost} valid queries were excluded because all their paired positives were removed"
            #         )
            # --------------------------------------------------

            filtered_qrels = filter_qrels_by_length(
                unique_queries_dataset.removed_ids,
                corpus_dataset.removed_ids,  # All removed corpus IDs (including positives)
                data_split["qrels"],
            )
            dist.barrier()

            # Validate filtered pairs using PyArrow — avoids materialising
            # millions of Python strings just for a subset check.

            corpus_ids_arr = corpus_dataset.data.table.column("id")
            query_ids_arr = unique_queries_dataset.data.table.column("id")
            qrel_qids = filtered_qrels.data.table.column("query_id")
            qrel_pids = filtered_qrels.data.table.column("positive_id")

            assert pc.all(
                pc.is_in(qrel_qids, value_set=pc.unique(query_ids_arr).combine_chunks())
            ).as_py(), "filtered qrels contain query IDs not in unique queries"
            assert pc.all(
                pc.is_in(
                    qrel_pids, value_set=pc.unique(corpus_ids_arr).combine_chunks()
                )
            ).as_py(), "filtered qrels contain positive IDs not in corpus"
            dist.barrier()

            if self.rank == 0:
                print(
                    f"full queries and corpus filtered in {(time.time()-start)/60:.2f}min"
                )
                num_queries_lost = (
                    pc.count_distinct(query_ids_arr).as_py()
                    - pc.count_distinct(qrel_qids).as_py()
                )
                if num_queries_lost > 0:
                    print(
                        f"Note: {num_queries_lost} valid queries were excluded because all their paired positives were removed"
                    )

        dataset = {
            "qrels": filtered_qrels,
            "unique_queries": unique_queries_dataset,
            "corpus": corpus_dataset,
            "n_positives": n_positives,
        }

        dist.barrier()
        if self.rank == 0:
            print(
                f"\nnumber unique queries: {return_formatted(len(unique_queries_dataset))}"
            )
            print(f"number of positives in corpus: {return_formatted(n_positives)}")
            print(f"number of documents: {return_formatted(len(corpus_dataset))}")

            print(
                f"total qrels pairs (with repetitions): {return_formatted(len(dataset['qrels']))}"
            )

        return dataset, corpus_dict

    def mine_one(
        self,
        dataset,
        model,
        has_title,
        corpus_dict,
        batch_size=64,
        top_k=100,
    ):

        sampler_queries = LenghtSortedSampler(dataset["unique_queries"])
        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
            tokenizer=self.tokenizer,
            eot_id=self.eot_id,
            add_special_tokens=self.add_special_tokens,
        )
        queries_loader = DataLoader(
            dataset["unique_queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=max(1, len(os.sched_getaffinity(0)) // 2 - 2),
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Decide whether query embeddings should live on CPU.
        # After all_gather every GPU holds ALL query embeddings; if the
        # gathered tensor + top-k bookkeeping would exceed 50 % of free
        # GPU memory, we stream embeddings to CPU during encoding and
        # keep them there throughout the search.
        n_queries = len(dataset["unique_queries"])
        hidden_size = _get_hidden_size(model)
        query_emb_bytes = n_queries * hidden_size * 4  # float32
        topk_bytes = n_queries * top_k * 12  # float32 scores + int64 indices
        free_mem, _ = torch.cuda.mem_get_info()
        # stream_to_cpu = (query_emb_bytes + topk_bytes) > 0.5 * free_mem
        stream_to_cpu = False
        # Broadcast stream_to_cpu from rank 0 so all GPUs take the same
        # code path — free_mem can differ across GPUs and a mismatch
        # would desync the NCCL collective sequence.
        _flag = torch.tensor([int(stream_to_cpu)], device=f"cuda:{self.rank}")
        dist.broadcast(_flag, src=0)
        stream_to_cpu = bool(_flag.item())
        del _flag

        dist.barrier()
        if self.rank == 0:
            start = time.time()
            print("\nbuilding query embeddings")
            if stream_to_cpu:
                print(
                    f"CPU mode: query embeddings ({query_emb_bytes / 2**30:.1f} GB) + "
                    f"top-k ({topk_bytes / 2**30:.1f} GB) exceed 50% of free GPU "
                    f"memory ({free_mem / 2**30:.1f} GB) — streaming to CPU"
                )
        query_embeddings = encode(
            model,
            queries_loader,
            prompt_type=PromptType.query,
            world_size=self.world_size,
            stream_to_cpu=stream_to_cpu,
        )

        dist.barrier()
        if self.rank == 0:
            print(f"queries embedding duration: {(time.time()-start)/60:.2f} min")

        # Create mappings from IDs to embedding indices
        # Extract columns from Arrow once — avoids repeated materialisation
        # of 14M Python strings each time dataset[...]["col"] is called.
        # unique_query_id_to_idx = {
        #     qid: idx for idx, qid in enumerate(dataset["unique_queries"]["id"])
        # }
        _unique_query_ids = dataset["unique_queries"].data.table.column("id").to_pylist()
        _qrel_query_ids = dataset["qrels"].data.table.column("query_id").to_pylist()
        _qrel_positive_ids = dataset["qrels"].data.table.column("positive_id").to_pylist()
        _corpus_ids = dataset["corpus"].data.table.column("id").to_pylist()
        unique_query_id_to_idx = {qid: idx for idx, qid in enumerate(_unique_query_ids)}

        dist.barrier()
        if self.rank == 0:
            print("\nQuery-positive scores will be computed during corpus search")
            print_memory_consumed(rank=self.rank)

        dist.barrier()
        chunk_size, query_chunk_size = estimate_chunk_sizes(query_embeddings)

        # Broadcast chunk sizes from rank 0 so the search loop has the
        # same number of iterations on every GPU (free_mem can differ).
        _cs = torch.tensor(
            [chunk_size, query_chunk_size], dtype=torch.long, device=f"cuda:{self.rank}"
        )
        dist.broadcast(_cs, src=0)
        chunk_size, query_chunk_size = int(_cs[0].item()), int(_cs[1].item())
        del _cs

        if self.rank == 0:
            print("\nBuilding document embeddings and computing query-positive scores")
            if query_chunk_size < query_embeddings.shape[0]:
                print(
                    f"Query chunking enabled: processing {return_formatted(query_embeddings.shape[0])} "
                    f"queries in chunks of {return_formatted(query_chunk_size)}"
                )

        start = time.time()
        top_scores, top_indices, query_positive_scores = search(
            model=model,
            query_embeddings=query_embeddings,
            corpus_dataset=dataset["corpus"],
            collate_fn=collate_fn,
            n_positives=dataset["n_positives"],
            # Re-use pre-extracted lists instead of dataset[...]["col"]
            # qrels_query_ids=dataset["qrels"]["query_id"],
            # qrels_positive_ids=dataset["qrels"]["positive_id"],
            # unique_query_ids=dataset["unique_queries"]["id"],
            # corpus_ids=dataset["corpus"]["id"],
            qrels_query_ids=_qrel_query_ids,
            qrels_positive_ids=_qrel_positive_ids,
            unique_query_ids=_unique_query_ids,
            unique_query_id_to_idx=unique_query_id_to_idx,
            corpus_ids=_corpus_ids,
            top_k=top_k,
            batch_size=batch_size,
            chunk_size=chunk_size,
            query_chunk_size=query_chunk_size,
        )

        del query_embeddings
        torch.cuda.empty_cache()

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60:.2f} min")
            print("\nbuilding negative lists")

        start = time.time()
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        hard_negatives, stats = self.get_hard_negatives(
            top_scores=top_scores,
            top_indices=top_indices,
            # Re-use pre-extracted lists instead of dataset[...]["col"]
            # corpus_ids=dataset["corpus"]["id"],
            # query_ids=dataset["qrels"]["query_id"],
            # positive_ids=dataset["qrels"]["positive_id"],
            # unique_query_ids=dataset["unique_queries"]["id"],
            corpus_ids=_corpus_ids,
            query_ids=_qrel_query_ids,
            positive_ids=_qrel_positive_ids,
            unique_query_ids=_unique_query_ids,
            query_positive_scores=query_positive_scores,
            unique_query_id_to_idx=unique_query_id_to_idx,
            has_title=has_title,
            corpus_dict=corpus_dict,
        )
        stats.total_queries = len(dataset["qrels"])

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60:.2f} min")

        return hard_negatives, stats, chunk_size

    def get_hard_negatives(
        self,
        top_scores,
        top_indices,
        corpus_ids,
        query_ids,
        positive_ids,
        unique_query_ids,
        query_positive_scores,
        unique_query_id_to_idx,
        has_title,
        corpus_dict,
    ):

        array_ids = np.asarray(corpus_ids)
        unique_query_ids = np.asarray(unique_query_ids)
        total_queries = len(query_ids)

        assert (
            top_scores.shape == top_indices.shape
        ), f"Scores / indices shape mismatch {top_scores.shape} {top_indices.shape}"
        assert (
            len(unique_query_ids) == top_scores.shape[0]
        ), f"Query count mismatch {len(unique_query_ids)} {top_scores.shape[0]}"
        assert (
            len(query_positive_scores) == total_queries
        ), f"Positive score mismatch {len(query_positive_scores)} {total_queries}"

        upper_thresholds_relevant_docs = min(100, int(0.1 * len(corpus_ids)))

        stats = TripletStats()
        hard_negatives = {}

        # Process each (query, positive) pair in qrels
        for qrel_idx, (q_id, p_id) in enumerate(zip(query_ids, positive_ids)):
            # Get the unique query index for this qrel entry
            unique_q_idx = unique_query_id_to_idx[q_id]

            # Get candidate documents for this query
            candidate_indices = top_indices[
                unique_q_idx, 5:upper_thresholds_relevant_docs
            ]
            candidate_scores = top_scores[
                unique_q_idx, 5:upper_thresholds_relevant_docs
            ]

            # Threshold based on this specific (query, positive) pair's score
            upper_threshold = min(0.9 * query_positive_scores[qrel_idx], 0.85)

            # Find valid hard negatives
            valid_mask = candidate_scores < upper_threshold
            valid_idx = np.where(valid_mask)[0]

            stats.update(valid_idx.size, 1)

            # Use composite key (query_id, positive_id) to store negatives per qrels entry
            key = (q_id, p_id)

            if valid_idx.size < 15:
                # Discard entries with fewer than 15 hard negatives —
                # they provide too little contrastive signal for training.
                continue

            # Cap number of negatives
            selected = valid_idx[:15]
            corpus_indices = candidate_indices[selected]

            neg_corpus_ids = array_ids[corpus_indices].tolist()

            if has_title:
                hard_negatives[key] = {
                    "id": neg_corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in neg_corpus_ids],
                    "title": [corpus_dict[id_]["title"] for id_ in neg_corpus_ids],
                }

            else:
                hard_negatives[key] = {
                    "id": neg_corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in neg_corpus_ids],
                }

            # DRAMATIC DROP IN PERFORMANCE FOR THE INDEXING OF CORPUS TEXT LIST
            # hard_negatives[q_id] = {
            #     "id": corpus_ids,
            #     "text": [corpus_texts[index] for index in corpus_indices],
            # }

        if self.rank == 0:
            print("total queries:", total_queries)
            print(
                f"{stats.less_than_24} examples have less than 24 hard negatives, {stats.less_than_24/total_queries*100: .2f}%, \
                    {stats.empty_entries/total_queries*100: .2f}% are empty"
            )

        return hard_negatives, stats

    def _mine_and_save_iterative(
        self,
        dataset,
        model,
        has_title,
        corpus_dict,
        query_dict,
        save_path,
        batch_size,
        subtask,
        has_subtasks,
    ):
        """Mine hard negatives for very large query sets in chunks.

        When unique queries exceed ``self.iterative_encode_threshold`` the full
        encode → search → get_hard_negatives → save pipeline is run on
        successive chunks of queries that individually fit in GPU memory.
        Each chunk produces its own Parquet shard so that peak RAM stays
        bounded throughout.

        Returns:
            TripletStats accumulated across all chunks.
        """
        n_queries = len(dataset["unique_queries"])
        chunk_size = self.iterative_encode_threshold
        n_chunks = (n_queries + chunk_size - 1) // chunk_size

        # -----------------------
        max_nreps = max(1, int(10**6 // self.iterative_encode_threshold))

        if self.rank == 0:
            print(
                f"\nIterative mining: {return_formatted(n_queries)} queries "
                f"in {n_chunks} chunks of up to "
                f"{return_formatted(chunk_size)}"
            )

        accumulated_stats = TripletStats()

        # Pre-extract qrels columns via Arrow — avoids materialising 14M
        # Python strings through HF Dataset.__getitem__ overhead.
        # all_qrel_query_ids = list(dataset["qrels"]["query_id"])
        # all_qrel_positive_ids = list(dataset["qrels"]["positive_id"])
        all_qrel_query_ids = dataset["qrels"].data.table.column("query_id").to_pylist()
        all_qrel_positive_ids = dataset["qrels"].data.table.column("positive_id").to_pylist()

        time_perf = {}
        start = time.time()
        for it, ci in enumerate(range(n_chunks)):
            if it >= max_nreps:
                break

            chunk_start = ci * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_queries)

            if self.rank == 0:
                print(
                    f"\n{'=' * 60}\n"
                    f"Iterative chunk {ci + 1}/{n_chunks}: "
                    f"queries {return_formatted(chunk_start)}"
                    f"–{return_formatted(chunk_end)}\n"
                    f"{'=' * 60}"
                )

            # Select unique queries for this chunk
            chunk_unique_queries = dataset["unique_queries"].select(
                range(chunk_start, chunk_end)
            )
            chunk_query_id_set = set(chunk_unique_queries["id"])

            # Filter qrels to only include queries in this chunk
            chunk_qrel_qids = []
            chunk_qrel_pids = []
            for qid, pid in zip(all_qrel_query_ids, all_qrel_positive_ids):
                if qid in chunk_query_id_set:
                    chunk_qrel_qids.append(qid)
                    chunk_qrel_pids.append(pid)

            chunk_qrels = Dataset.from_dict(
                {"query_id": chunk_qrel_qids, "positive_id": chunk_qrel_pids}
            )

            if self.rank == 0:
                print(
                    f"chunk unique queries: {return_formatted(len(chunk_unique_queries))}, "
                    f"chunk qrels: {return_formatted(len(chunk_qrels))}"
                )

            chunk_dataset = {
                "qrels": chunk_qrels,
                "unique_queries": chunk_unique_queries,
                "corpus": dataset["corpus"],
                "n_positives": dataset["n_positives"],
            }

            # mine_one will see ≤ self.iterative_encode_threshold queries,
            # so it will use the normal GPU path (no stream_to_cpu).
            hard_negatives, stats, chunk_size_doc = self.mine_one(
                dataset=chunk_dataset,
                model=model,
                batch_size=batch_size,
                has_title=has_title,
                corpus_dict=corpus_dict,
            )
            stats.total_queries = len(chunk_qrels)
            accumulated_stats.merge(stats)

            dist.barrier()
            if self.rank == 0:
                print(f"\n\nupdating dataset dict for chunk {ci + 1}/{n_chunks}")

            # Build dataset dict and save shard immediately
            dataset_dict = {}
            update_dataset_dict(
                dataset_dict=dataset_dict,
                qrels=chunk_qrels,
                negatives=hard_negatives,
                has_title=has_title,
                corpus_dict=corpus_dict,
                query_dict=query_dict,
                subtask=subtask if has_subtasks else None,
            )

            # Shard naming: append chunk index to distinguish from
            # single-chunk shards produced by the non-iterative path.
            base_name = subtask if subtask is not None else "data"
            shard_name = f"{base_name}_chunk_{ci}"

            dist.barrier()
            if self.rank == 0:
                print(f"\nsaving shard {shard_name}")

            save_dataset_shard_to_parquet(
                dataset_dict=dataset_dict,
                save_dir=save_path,
                shard_name=shard_name,
                has_title=has_title,
                rank=self.rank,
            )

            del hard_negatives, chunk_dataset, dataset_dict
            del chunk_qrels, chunk_unique_queries
            del chunk_qrel_qids, chunk_qrel_pids
            torch.cuda.empty_cache()

            dist.barrier()
            torch.cuda.synchronize()
            if self.rank == 0:
                elapsed_time = time.time() - start

                time_perf[f"iter_{it}_time_min"] = elapsed_time / 60
                time_perf["chunk_size"] = chunk_size_doc

                with open(
                    f"time_to_1M_chunk-size{self.iterative_encode_threshold}.json", "w"
                ) as f:
                    json.dump(time_perf, f, indent=2)

                print(f"Chunk {ci + 1}/{n_chunks} done")
                print_memory_consumed(rank=self.rank)

        return accumulated_stats

    def mine_negatives(self, model, batch_size=64):

        self.batch_size = batch_size
        for task_name in self.task_names:

            save_path, category = get_category_path(task_name, self.path)

            if self.rank == 0:
                print(f"\n\nLOADING DATASET {category}: {task_name}\n")

            task = get_task(task_name)
            subtasks = getattr(task, "subtasks", None)
            has_subtasks = subtasks is not None
            if not has_subtasks:
                subtasks = [None]

            accumulated_stats = TripletStats()
            per_subset_stats = {} if has_subtasks else None

            for subtask in subtasks:
                if self.rank == 0 and subtask is not None:
                    print(f"\n--- subtask: {subtask} ---")

                (
                    data_split,
                    corpus_dict,
                    query_dict,
                    has_title,
                    n_positives,
                ) = load_task_data(task, subtask)

                if self.rank == 0:
                    print(f"\n\nPREPARING DATASET {category}: {task_name}\n")

                dataset, corpus_dict = self.prepare_dataset(
                    data_split=data_split,
                    corpus_dict=corpus_dict,
                    task_metadata=task.metadata,
                    n_positives=n_positives,
                )

                dist.barrier()
                if self.rank == 0:
                    print(f"\n\nprocessing dataset {task_name}")
                    print_memory_consumed(rank=self.rank)

                # Decide whether to use the iterative path for very
                # large query sets (> self.iterative_encode_threshold).
                n_unique_queries = len(dataset["unique_queries"])
                use_iterative = n_unique_queries > self.iterative_encode_threshold
                # Broadcast from rank 0 so every GPU takes the same path.
                _flag = torch.tensor([int(use_iterative)], device=f"cuda:{self.rank}")
                dist.broadcast(_flag, src=0)
                use_iterative = bool(_flag.item())
                del _flag

                if use_iterative:
                    chunk_stats = self._mine_and_save_iterative(
                        dataset=dataset,
                        model=model,
                        has_title=has_title,
                        corpus_dict=corpus_dict,
                        query_dict=query_dict,
                        save_path=save_path,
                        batch_size=batch_size,
                        subtask=subtask,
                        has_subtasks=has_subtasks,
                    )
                    accumulated_stats.merge(chunk_stats)
                    if has_subtasks:
                        per_subset_stats[subtask] = chunk_stats

                    del dataset, data_split
                    torch.cuda.empty_cache()
                    print_memory_consumed(rank=self.rank)
                    dist.barrier()
                    continue

                # --------------- normal (non-iterative) path ---------------
                hard_negatives, stats, _ = self.mine_one(
                    dataset=dataset,
                    model=model,
                    batch_size=batch_size,
                    has_title=has_title,
                    corpus_dict=corpus_dict,
                )
                stats.total_queries = len(dataset["qrels"])
                accumulated_stats.merge(stats)
                if has_subtasks:
                    per_subset_stats[subtask] = stats

                dist.barrier()
                if self.rank == 0:
                    print(
                        f"\n\nupdating dataset dict for {task_name}"
                        + (f" subtask {subtask}" if subtask else "")
                    )

                # Build a fresh dict for this subtask only — avoids
                # accumulating all subtasks in memory (critical for
                # StackExchange with 174 subtasks / ~4.7 M rows).
                dataset_dict = {}
                update_dataset_dict(
                    dataset_dict=dataset_dict,
                    qrels=dataset["qrels"],
                    negatives=hard_negatives,
                    has_title=has_title,
                    corpus_dict=corpus_dict,
                    query_dict=query_dict,
                    subtask=subtask if has_subtasks else None,
                )

                # Free per-subtask objects before saving
                del hard_negatives, dataset, data_split

                dist.barrier()
                if self.rank == 0:
                    print(
                        f"\n\nsaving shard for {task_name}"
                        + (f" subtask {subtask}" if subtask else "")
                    )

                # Save this subtask as a compressed Parquet shard,
                # then release the dict so memory stays bounded.
                shard_name = subtask if subtask is not None else "data"
                save_dataset_shard_to_parquet(
                    dataset_dict=dataset_dict,
                    save_dir=save_path,
                    shard_name=shard_name,
                    has_title=has_title,
                    rank=self.rank,
                )
                del dataset_dict

                torch.cuda.empty_cache()
                print_memory_consumed(rank=self.rank)
                dist.barrier()

            # Write metadata once after all subtasks are done
            save_dataset_metadata(
                save_dir=save_path,
                stats=accumulated_stats,
                model_name=self.model_name,
                rank=self.rank,
                per_subset_stats=per_subset_stats,
            )
