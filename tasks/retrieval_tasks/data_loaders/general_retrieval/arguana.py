from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
from typing import Set
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def get_mteb_arguana_texts() -> tuple[Set[str], Set[str]]:
    """
    Load mteb/arguana evaluation dataset and return sets of normalized
    query texts and corpus texts for deduplication.
    """
    # Load MTEB arguana corpus and queries
    corpus = load_dataset("mteb/arguana", name="corpus", split="corpus")
    queries = load_dataset("mteb/arguana", name="queries", split="queries")

    # Build sets of normalized texts
    corpus_texts = {normalize_text(row["text"]) for row in corpus}
    query_texts = {normalize_text(row["text"]) for row in queries}

    return query_texts, corpus_texts


def clear_arguana_overlap(
    dataset,
    anchor_field: str,
    positive_field: str,
    mteb_query_texts: Set[str],
    mteb_corpus_texts: Set[str],
):
    """
    Filter BeIR/arguana-generated-queries dataset to remove examples
    that overlap with mteb/arguana evaluation set.
    """

    def is_not_overlapping(example):
        query_norm = normalize_text(example[anchor_field])
        positive_norm = normalize_text(example[positive_field])

        # Remove if query OR positive text appears in MTEB evaluation set
        query_overlaps = query_norm in mteb_query_texts
        positive_overlaps = positive_norm in mteb_corpus_texts

        return not (query_overlaps or positive_overlaps)

    filtered_dataset = dataset.filter(is_not_overlapping)
    return filtered_dataset


def load_arguana_dedup_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """Load BeIR/arguana-generated-queries with deduplication against mteb/arguana using vectorized operations.

    This removes any query-positive pairs where either the query or positive text
    appears in the mteb/arguana evaluation set, preventing train-test contamination.
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset...")
    
    # Load the BeIR arguana dataset
    dataset = load_dataset(task.hf_name, split=task.split)
    
    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(len(dataset))}")
        start = time.time()
        print("Loading mteb/arguana for deduplication...")

    # Get MTEB arguana texts for deduplication
    mteb_query_texts, mteb_corpus_texts = get_mteb_arguana_texts()
    
    dist.barrier()
    if rank == 0:
        print(
            f"Found {return_formatted(len(mteb_query_texts))} MTEB queries and {return_formatted(len(mteb_corpus_texts))} MTEB corpus texts"
        )
        print("Filtering overlapping examples...")

    # Filter out overlapping examples
    original_size = len(dataset)
    dataset = clear_arguana_overlap(
        dataset,
        task.anchor_name,
        task.positive_name,
        mteb_query_texts,
        mteb_corpus_texts,
    )
    filtered_size = len(dataset)
    
    dist.barrier()
    if rank == 0:
        print(
            f"Removed {original_size - filtered_size} overlapping examples ({original_size} -> {filtered_size})"
        )
        start = time.time()
        print("Converting to pandas...")

    # Convert to pandas DataFrame
    df = dataset.select_columns([task.anchor_name, task.positive_name]).to_pandas()
    df.columns = ["query", "positive"]
    n_pairs = len(df)
    
    dist.barrier()
    if rank == 0:
        print(f"Conversion done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Building query-positive pairs with deduplication...")
    
    # Get unique queries using pandas
    unique_query_mask = ~df["query"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_texts = df.loc[unique_query_mask, "query"].tolist()
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]
    
    # Create query text to ID mapping
    query_text_to_id = pd.Series(unique_query_ids, index=df.loc[unique_query_mask, "query"].values)
    
    # Map all queries to their unique IDs
    query_ids = df["query"].map(query_text_to_id).tolist()
    
    # Get unique positives
    unique_positive_mask = ~df["positive"].duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index.values
    unique_positive_texts = df.loc[unique_positive_mask, "positive"].tolist()
    unique_positive_ids = [f"doc_{i}" for i in unique_positive_idx]
    
    # Create document text to ID mapping
    doc_text_to_id = pd.Series(unique_positive_ids, index=df.loc[unique_positive_mask, "positive"].values)
    
    # Map all positives to their unique IDs
    positive_ids = df["positive"].map(doc_text_to_id).tolist()
    
    dist.barrier()
    if rank == 0:
        print(f"Deduplication done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(unique_query_ids))} unique queries before limiting")
        print(f"Found {return_formatted(len(unique_positive_ids))} unique documents")
        start = time.time()
        print("Applying query limiting...")
    
    # Apply query limiting if needed
    if max_num_queries is not None and len(unique_query_idx) > max_num_queries:
        if rank == 0:
            print(
                f"Number of unique queries {return_formatted(len(unique_query_idx))} > {max_num_queries//10**6}M: limiting queries"
            )
        
        unique_query_texts = unique_query_texts[:max_num_queries]
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_idx = unique_query_idx[:max_num_queries]
        
        (
            query_ids,
            positive_ids,
            document_ids,
            document_texts,
            _,
            n_positives,
        ) = limit_number_of_queries(
            query_ids=query_ids,
            positive_ids=positive_ids,
            unique_query_idx=unique_query_idx,
            n_pairs=n_pairs,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=None,
            has_title=False,
            max_queries=max_num_queries,
        )
        
        if rank == 0:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
            print(f"Positives referenced by filtered pairs: {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")
    else:
        # No limiting needed
        document_ids = unique_positive_ids
        document_texts = unique_positive_texts
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(
                f"Found {return_formatted(len(unique_query_ids))} unique queries (under {max_num_queries//10**6}M limit)"
            )
            print(f"Total number of query-positive pairs: {return_formatted(len(query_ids))}")
            print(f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # Assertions to ensure data consistency
    assert set(positive_ids).issubset(
        set(document_ids)
    ), "filtered qrels contain positive IDs not in document list"
    
    assert set(unique_positive_ids).issubset(
        set(document_ids)
    ), "unique positives not in document list"
    
    assert set(corpus_dict.keys()) == set(document_ids), "corpus_dict keys mismatch with document_ids"

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=False,
        n_positives=n_positives,
    )


class Arguana(AbsTask):
    """BeIR Arguana dataset with deduplication against mteb/arguana eval set."""

    language = "en"

    hf_name = "BeIR/arguana-generated-queries"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    negative_name = "negative"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["ArguAna"]})
    loader = load_arguana_dedup_retrieval
