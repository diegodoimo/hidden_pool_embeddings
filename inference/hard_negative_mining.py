import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from inference.create_datasets import create_dataset, filter_qrels_by_length

from functools import partial
import numpy as np
import torch.distributed as dist

from datasets import Dataset, Features, Value, Sequence
from tasks.load_datasets import load_task_data
from tasks import get_task
from tasks.task_categories import get_category_path
from utils.sorted_sampler import LenghtSortedSampler
from pathlib import Path
import time
from .helpers import encode, search, collate_fn_with_padding
from collections import Counter
from dataclasses import dataclass
from datasets import DatasetInfo
import json
from utils.helpers import print_memory_consumed, return_formatted


def estimate_chunk_size(query_embeddings, max_chunk=5 * 10**4):
    free_mem, _ = torch.cuda.mem_get_info()
    bytes_per_number = query_embeddings.element_size()
    bytes_per_doc = query_embeddings.shape[1] * bytes_per_number
    bytes_per_sim_column = query_embeddings.shape[0] * bytes_per_number
    chunk = int(0.8 * free_mem // (bytes_per_doc + bytes_per_sim_column))
    return max(1000, min(chunk, max_chunk))


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
    n = len(query_ids)

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

    # Use generators to avoid temporary list copies
    dataset_dict["query_text"].extend(query_dict[qid]["text"] for qid in query_ids)
    dataset_dict["query_id"].extend(query_ids)
    dataset_dict["positive_text"].extend(
        corpus_dict[pid]["text"] for pid in positive_ids
    )
    dataset_dict["positive_id"].extend(positive_ids)
    dataset_dict["negative_text"].extend(
        negatives[(q, p)]["text"] for q, p in zip(query_ids, positive_ids)
    )
    dataset_dict["negative_id"].extend(
        negatives[(q, p)]["id"] for q, p in zip(query_ids, positive_ids)
    )
    if has_title:
        dataset_dict["positive_title"].extend(
            corpus_dict[pid]["title"] for pid in positive_ids
        )
        dataset_dict["negative_title"].extend(
            negatives[(q, p)]["title"] for q, p in zip(query_ids, positive_ids)
        )
    if subtask is not None:
        dataset_dict["subset"].extend([subtask] * n)


def save_dataset_dict_to_disk(
    dataset_dict,
    save_path,
    stats,
    model_name,
    has_title,
    rank,
    per_subset_stats=None,
):
    """Convert *dataset_dict* to an HF ``Dataset`` and save to *save_path*.

    Only rank-0 performs the I/O.  The ``Dataset`` object is deleted
    immediately after saving to keep peak memory low (important when the
    dict contains up to ~10 M rows).
    """
    if rank != 0:
        return

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
    if "subset" in dataset_dict:
        features_dict["subset"] = Value("string")

    dataset = Dataset.from_dict(dataset_dict, features=Features(features_dict))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)
    del dataset  # free Arrow memory immediately

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
    with open(f"{save_path}/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = task_names
        self.padding_side = padding_side
        self.max_length = max_length

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
        if corpus_dataset.removed_ids > 0:
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
            filtered_qrels = filter_qrels_by_length(
                unique_queries_dataset.removed_ids,
                corpus_dataset.removed_ids,  # All removed corpus IDs (including positives)
                data_split["qrels"],
            )

            corpus_ids_set = set(corpus_dataset["id"])
            # Check that filtered pairs only contain valid IDs
            assert set(filtered_qrels["query_id"]).issubset(
                set(unique_queries_dataset["id"])
            ), "filtered qrels contain query IDs not in unique queries"
            assert set(filtered_qrels["positive_id"]).issubset(
                corpus_ids_set
            ), "filtered qrels contain positive IDs not in corpus"

            if self.rank == 0:
                print(
                    f"full queries and corpus filters in {(time.time()-start)/60:.2f}min"
                )
                num_queries_lost = len(set(unique_queries_dataset["id"])) - len(
                    set(filtered_qrels["query_id"])
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
            eot_id=self.tokenizer.pad_token_id,
        )
        queries_loader = DataLoader(
            dataset["unique_queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        dist.barrier()
        if self.rank == 0:
            start = time.time()
            print("\nbuilding query embeddings")

        query_embeddings = encode(
            model,
            queries_loader,
            prompt_type=PromptType.query,
            world_size=self.world_size,
        )

        dist.barrier()
        if self.rank == 0:
            print(f"queries embedding duration: {(time.time()-start)/60:.2f} min")

        # Create mappings from IDs to embedding indices
        unique_query_id_to_idx = {
            qid: idx for idx, qid in enumerate(dataset["unique_queries"]["id"])
        }

        dist.barrier()
        if self.rank == 0:
            print("\nQuery-positive scores will be computed during corpus search")
            print_memory_consumed(rank=self.rank)

        dist.barrier()
        chunk_size = estimate_chunk_size(query_embeddings)
        if self.rank == 0:
            print("\nBuilding document embeddings and computing query-positive scores")

        start = time.time()
        top_scores, top_indices, query_positive_scores = search(
            model=model,
            query_embeddings=query_embeddings,
            corpus_dataset=dataset["corpus"],
            collate_fn=collate_fn,
            n_positives=dataset["n_positives"],
            qrels_query_ids=dataset["qrels"]["query_id"],
            qrels_positive_ids=dataset["qrels"]["positive_id"],
            unique_query_ids=dataset["unique_queries"]["id"],
            unique_query_id_to_idx=unique_query_id_to_idx,
            corpus_ids=dataset["corpus"]["id"],
            top_k=top_k,
            batch_size=batch_size,
            chunk_size=chunk_size,
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
            corpus_ids=dataset["corpus"]["id"],
            query_ids=dataset["qrels"]["query_id"],
            positive_ids=dataset["qrels"]["positive_id"],
            unique_query_ids=dataset["unique_queries"]["id"],
            query_positive_scores=query_positive_scores,
            unique_query_id_to_idx=unique_query_id_to_idx,
            has_title=has_title,
            corpus_dict=corpus_dict,
        )
        stats.total_queries = len(dataset["qrels"])

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60:.2f} min")

        return hard_negatives, stats

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

        upper_thresholds_relevent_docs = min(150, int(0.1 * len(corpus_ids)))

        stats = TripletStats()
        hard_negatives = {}

        # Process each (query, positive) pair in qrels
        for qrel_idx, (q_id, p_id) in enumerate(zip(query_ids, positive_ids)):
            # Get the unique query index for this qrel entry
            unique_q_idx = unique_query_id_to_idx[q_id]

            # Get candidate documents for this query
            candidate_indices = top_indices[
                unique_q_idx, 5:upper_thresholds_relevent_docs
            ]
            candidate_scores = top_scores[
                unique_q_idx, 5:upper_thresholds_relevent_docs
            ]

            # Threshold based on this specific (query, positive) pair's score
            upper_threshold = min(0.95 * query_positive_scores[qrel_idx], 0.9)

            # Find valid hard negatives
            valid_mask = candidate_scores < upper_threshold
            valid_idx = np.where(valid_mask)[0]

            stats.update(valid_idx.size, 1)

            # Use composite key (query_id, positive_id) to store negatives per qrels entry
            key = (q_id, p_id)

            if valid_idx.size == 0:
                # Use the same structure as the normal case, just with empty lists
                if has_title:
                    hard_negatives[key] = {"id": [], "text": [], "title": []}
                else:
                    hard_negatives[key] = {"id": [], "text": []}
                continue

            # Cap number of negatives
            selected = valid_idx[:24]
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

            dataset_dict = {}
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

                hard_negatives, stats = self.mine_one(
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
                    print(f"\n\nsaving dataset {task_name}")

                save_dataset_dict_to_disk(
                    dataset_dict=dataset_dict,
                    save_path=save_path,
                    stats=accumulated_stats,
                    model_name=self.model_name,
                    has_title=has_title,
                    rank=self.rank,
                    per_subset_stats=per_subset_stats,
                )

                torch.cuda.empty_cache()
                print_memory_consumed(rank=self.rank)

            del dataset_dict

    def save_to_disk(
        self,
        qrels,
        negatives,
        has_title,
        stats,
        save_path,
        corpus_dict,
        query_dict,
    ):

        # Retrieve texts from corpus_dict using IDs from qrels
        query_ids = qrels["query_id"]
        positive_ids = qrels["positive_id"]
        query_texts = [query_dict[qid]["text"] for qid in query_ids]
        positive_text = [corpus_dict[pid]["text"] for pid in positive_ids]

        # Use (query_id, positive_id) tuples to get negatives for each qrels entry
        negative_text = [
            negatives[(q_id, p_id)]["text"]
            for q_id, p_id in zip(query_ids, positive_ids)
        ]
        negative_id = [
            negatives[(q_id, p_id)]["id"] for q_id, p_id in zip(query_ids, positive_ids)
        ]

        if has_title:
            negative_title = [
                negatives[(q_id, p_id)]["title"]
                for q_id, p_id in zip(query_ids, positive_ids)
            ]
            positive_title = [corpus_dict[pid]["title"] for pid in positive_ids]

            dataset = Dataset.from_dict(
                {
                    "query_text": query_texts,
                    "query_id": query_ids,
                    "positive_text": positive_text,
                    "positive_title": positive_title,
                    "positive_id": positive_ids,
                    "negative_text": negative_text,
                    "negative_title": negative_title,
                    "negative_id": negative_id,
                },
                features=Features(
                    {
                        "query_text": Value("string"),
                        "query_id": Value("string"),
                        "positive_text": Value("string"),
                        "positive_title": Value("string"),
                        "positive_id": Value("string"),
                        "negative_text": Sequence(Value("string")),
                        "negative_title": Sequence(Value("string")),
                        "negative_id": Sequence(Value("string")),
                    }
                ),
            )
        else:
            dataset = Dataset.from_dict(
                {
                    "query_text": query_texts,
                    "query_id": query_ids,
                    "positive_text": positive_text,
                    "positive_id": positive_ids,
                    "negative_text": negative_text,
                    "negative_id": negative_id,
                },
                features=Features(
                    {
                        "query_text": Value("string"),
                        "query_id": Value("string"),
                        "positive_text": Value("string"),
                        "positive_id": Value("string"),
                        "negative_text": Sequence(Value("string")),
                        "negative_id": Sequence(Value("string")),
                    }
                ),
            )

        # Save metadata
        metadata = {
            "num_triples": stats.total_queries,
            "num_empty_negative_entries": stats.empty_entries,
            "num_with_7_hard_negatives": stats.total_queries - stats.less_than_7,
            "num_with_15_hard_negatives": stats.total_queries - stats.less_than_15,
            "num_with_24_hard_negatives": stats.total_queries - stats.less_than_24,
            "embedder": self.model_name,
        }

        # Get the categorized path for this task
        # save_path = get_category_path(task_name, self.path)

        dist.barrier()
        if self.rank == 0:
            # Create parent directories if they don't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(save_path)

            with open(f"{save_path}/dataset_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
