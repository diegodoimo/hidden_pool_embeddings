"""Data loader for F2LLM datasets saved as local parquet files.

These parquet files are produced by:
    python tokenize_data.py --f2llm --save_raw_data_only ...

Expected columns (written by _assign_dedup_ids in tokenize_data.py):
    query_text    : str
    positive_text : str
    negative_text : list[str]   – one list per row
    query_id      : str         – "query_<first_occurrence_idx>"
    positive_id   : str         – "doc_<first_occurrence_idx>"
    negative_id   : list[str]   – parallel to negative_text; shares "doc_*" namespace

The loader mirrors tasks/retrieval_loaders.py::from_one_hf_dataset step by step,
so the output RetrievalRawData is fully compatible with
inference/hard_negative_mining.HardNegativesMiner.
"""

import gc
import time
from dataclasses import dataclass
from typing import ClassVar, List, Optional

import pandas as pd
import torch.distributed as dist

from tasks.abs_task import TaskMetadata
from tasks.data_helpers import LazyCorpusDict, RetrievalRawData
from utils.helpers import return_formatted


def from_f2llm_parquet(
    task,
    max_num_queries: Optional[int] = None,
    rank: Optional[int] = None,
    subtask=None,
) -> RetrievalRawData:
    """Load F2LLM data from a local parquet file produced by tokenize_data.py.

    Mirrors retrieval_loaders.from_one_hf_dataset:
      1. Load parquet → pandas DataFrame; drop null rows.
      2. Reuse stored query_id / positive_id (already deduplicated by
         _assign_dedup_ids — identical texts carry the same ID).
      3. Collect unique negatives whose ID is not yet in the positive set
         (IDs share the same "doc_*" namespace, so set-membership is sufficient).
      4. Build document list with unique positives first (indices 0…n_positives-1)
         followed by extra negatives.
      5. Optionally limit to max_num_queries unique queries.
      6. Return RetrievalRawData with LazyCorpusDicts.
    """
    rank = dist.get_rank() if rank is None else rank

    if rank == 0:
        start = time.time()
        print(f"Loading F2LLM parquet: {task.parquet_path}")

    # save_raw_data in tokenize_data.py writes all string columns as
    # large_utf8 (64-bit offsets), so pd.read_parquet works correctly even
    # for large datasets like bioasq whose list-column values exceed 2 GB.
    df = pd.read_parquet(task.parquet_path)

    n_pairs_raw = len(df)
    verbose = n_pairs_raw > 10**6

    # ------------------------------------------------------------------
    # 1. Drop rows where query or positive text is null.
    # ------------------------------------------------------------------
    null_mask = df["query_text"].isna() | df["positive_text"].isna()
    if null_mask.any():
        n_null = int(null_mask.sum())
        if rank == 0:
            print(f"Dropping {n_null} rows with null query or positive text")
        df = df[~null_mask].reset_index(drop=True)

    n_pairs = len(df)

    dist.barrier()
    if rank == 0 and verbose:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(n_pairs)}")
        print("Finding unique queries and positives...")

    # ------------------------------------------------------------------
    # 2a. Queries — reuse stored query_id.
    #     Same text → same "query_*" ID (guaranteed by deduplicate in
    #     _assign_dedup_ids), so no re-hashing is needed.
    # ------------------------------------------------------------------
    query_ids: List[str] = df["query_id"].tolist()
    unique_q_mask = ~df["query_id"].duplicated(keep="first")
    unique_query_ids: List[str] = df.loc[unique_q_mask, "query_id"].tolist()
    unique_query_texts: List[str] = df.loc[unique_q_mask, "query_text"].tolist()

    # ------------------------------------------------------------------
    # 2b. Positives — reuse stored positive_id.
    #     positive_id and negative_id share the same "doc_*" namespace:
    #     if a negative text equals a positive text they carry the same ID.
    # ------------------------------------------------------------------
    positive_ids: List[str] = df["positive_id"].tolist()
    unique_pos_mask = ~df["positive_id"].duplicated(keep="first")
    unique_positive_ids: List[str] = df.loc[unique_pos_mask, "positive_id"].tolist()
    unique_positive_texts: List[str] = df.loc[unique_pos_mask, "positive_text"].tolist()
    n_positives = len(unique_positive_ids)

    # ------------------------------------------------------------------
    # 3. Negatives: unique "doc_*" IDs not already covered by positives.
    #    Explode list columns in parallel, deduplicate by ID, then filter.
    # ------------------------------------------------------------------
    pos_id_set = set(unique_positive_ids)

    neg_id_col = df["negative_id"].explode().reset_index(drop=True)
    neg_text_col = df["negative_text"].explode().reset_index(drop=True)
    neg_df = pd.DataFrame({"id": neg_id_col, "text": neg_text_col}).dropna(subset=["text"])
    neg_df = (
        neg_df
        .drop_duplicates(subset=["id"], keep="first")
        .loc[lambda d: ~d["id"].isin(pos_id_set)]
        .reset_index(drop=True)
    )
    neg_ids: List[str] = neg_df["id"].tolist()
    neg_texts: List[str] = neg_df["text"].tolist()

    del df, neg_df, neg_id_col, neg_text_col
    gc.collect()

    if rank == 0 and verbose:
        print(f"Found {return_formatted(len(neg_ids))} unique negatives not in positives")

    # ------------------------------------------------------------------
    # Sanity check before optional limiting.
    # ------------------------------------------------------------------
    assert set(positive_ids).issubset(set(unique_positive_ids)), (
        "positive IDs contain entries not found in unique positives — data integrity error"
    )

    # ------------------------------------------------------------------
    # 5. Optional query limiting (mirrors from_one_hf_dataset).
    # ------------------------------------------------------------------
    if max_num_queries is not None and len(unique_query_ids) > max_num_queries:
        if rank == 0:
            print(
                f"Limiting queries from {return_formatted(len(unique_query_ids))} "
                f"to {return_formatted(max_num_queries)}"
            )

        allowed_query_ids = set(unique_query_ids[:max_num_queries])
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_texts = unique_query_texts[:max_num_queries]

        # Keep only qrel pairs within the surviving query set.
        keep_pairs = [i for i, qid in enumerate(query_ids) if qid in allowed_query_ids]
        query_ids = [query_ids[i] for i in keep_pairs]
        positive_ids = [positive_ids[i] for i in keep_pairs]

        # Recompute which positives are still referenced by surviving pairs.
        referenced_pos = set(positive_ids)
        still_in = [uid in referenced_pos for uid in unique_positive_ids]
        unique_positive_ids = [uid for uid, m in zip(unique_positive_ids, still_in) if m]
        unique_positive_texts = [t for t, m in zip(unique_positive_texts, still_in) if m]
        n_positives = len(unique_positive_ids)

        # Re-filter negatives: an ID that moved from "extra" to "positive" after
        # limiting must be removed from the negative list to avoid duplication.
        pos_id_set = set(unique_positive_ids)
        keep_neg = [i for i, nid in enumerate(neg_ids) if nid not in pos_id_set]
        neg_ids = [neg_ids[i] for i in keep_neg]
        neg_texts = [neg_texts[i] for i in keep_neg]

    # ------------------------------------------------------------------
    # 4. Build document list: unique positives first, extra negatives after.
    # ------------------------------------------------------------------
    document_ids = list(unique_positive_ids) + neg_ids
    document_texts = list(unique_positive_texts) + neg_texts
    document_titles = None  # F2LLM parquets have no title column

    assert set(positive_ids).issubset(set(document_ids)), (
        "filtered qrels contain positive IDs not in the document list"
    )

    if rank == 0:
        print(f"Found {return_formatted(len(unique_query_ids))} unique queries")
        print(f"Total query-positive pairs: {return_formatted(len(query_ids))}")
        print(f"Positives in corpus (n_positives): {return_formatted(n_positives)}")
        print(f"Total documents in corpus: {return_formatted(len(document_ids))}")

    # ------------------------------------------------------------------
    # 6. Lightweight lookup dicts (no text duplication in RAM).
    # ------------------------------------------------------------------
    corpus_dict = LazyCorpusDict(ids=document_ids, texts=document_texts)
    query_dict = LazyCorpusDict(ids=unique_query_ids, texts=unique_query_texts)

    dist.barrier()

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        query_dict=query_dict,
        has_title=False,
        n_positives=n_positives,
    )


@dataclass
class F2LLMParquetTask:
    """Thin task descriptor for a single F2LLM dataset loaded from a local parquet.

    Compatible with tasks.load_datasets.load_task_data and
    inference.hard_negative_mining.HardNegativesMiner.

    Parameters
    ----------
    parquet_path : str
        Absolute path to the parquet written by:
            python tokenize_data.py --f2llm --save_raw_data_only ...
    metadata : TaskMetadata
        Must have type="Retrieval" and a prompt dict matching the target
        instruction template (e.g. TaskMetadata(type="Retrieval", prompt={...})).
        Use make_f2llm_task() to inherit the prompt from the registered task.

    Class-level attributes (not dataclass fields, shared by all instances):
        loader               : from_f2llm_parquet (called by load_task_data)
        has_multiple_datasets: False
        subtasks             : None  (single-split dataset)
        negative_name        : None
    """

    parquet_path: str
    metadata: TaskMetadata

    # Not annotated → treated as class attributes, not dataclass fields.
    # staticmethod prevents Python's descriptor protocol from binding the
    # instance as the first argument when loader_func is retrieved via
    # getattr(task, "loader"), which would cause "multiple values for
    # argument 'task'" when _load_retrieval_data calls loader_func(task=task, ...).
    loader = staticmethod(from_f2llm_parquet)
    has_multiple_datasets = False
    subtasks = None
    negative_name = None
    # Ensure load_task_data always routes through _load_retrieval_data, even
    # for tasks whose metadata.type is Classification / Clustering / STS / etc.
    # All F2LLM parquets share the same retrieval-style schema regardless of
    # the original task type.
    use_hard_negative_mining = True


def make_f2llm_task(ds_name: str, parquet_path: str) -> F2LLMParquetTask:
    """Factory: build an F2LLMParquetTask inheriting the prompt from the registered task.

    The instruction-template prompt stored in the original task class is reused
    so that encoding at mining time uses the same prompt as training.

    Parameters
    ----------
    ds_name : str
        Dataset name as registered in tasks.NAME_TO_TASK (e.g. "arguana").
    parquet_path : str
        Path to the corresponding parquet saved by tokenize_data.py.

    Returns
    -------
    F2LLMParquetTask
        Ready to be passed to load_task_data() or HardNegativesMiner.
    """
    from tasks import NAME_TO_TASK  # local import to avoid circular dependency

    original_task = NAME_TO_TASK.get(ds_name)
    if original_task is None:
        raise ValueError(
            f"Unknown dataset '{ds_name}'. Available: {sorted(NAME_TO_TASK.keys())}"
        )
    return F2LLMParquetTask(
        parquet_path=parquet_path,
        metadata=original_task.metadata,
    )
