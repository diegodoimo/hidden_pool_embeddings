"""
STS-specific loader functions.
These loaders handle Semantic Textual Similarity tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import time
import torch.distributed as dist
import pandas as pd
import numpy as np

from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


def load_sts_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """
    Load STS datasets for retrieval following Lee et al. (2025a) using vectorized operations.

    For textual similarity (STS) datasets, we construct a query-positive pair from any
    sentence pair whose similarity score is at least 4 and another pair with the query
    and positive switched. Hard negatives are mined from the corpus.

    Used by: STS12, STS22, STSBenchmark
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset...")
    
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)
    
    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(len(dataset))}")
        start = time.time()
        print("Converting to pandas and filtering by score...")

    score_name = getattr(task, "score_name", "score")
    score_threshold = getattr(task, "score_threshold", 4.0)

    # Convert to pandas DataFrame for vectorized operations
    cols_to_load = [task.query_name, task.positive_name, score_name]
    df = dataset.select_columns(cols_to_load).to_pandas()
    df.columns = ["sentence1", "sentence2", "score"]
    
    # Filter by score threshold
    df = df[df["score"] >= score_threshold].reset_index(drop=True)
    
    dist.barrier()
    if rank == 0:
        print(f"Filtering done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(df))} pairs above threshold")
        start = time.time()
        print("Building sentence pairs with switching...")
    
    # Create both directions: (s1, s2) and (s2, s1) for different sentences
    # For identical sentences, only keep one pair
    pairs_forward = df[["sentence1", "sentence2"]].copy()
    pairs_forward.columns = ["query", "positive"]
    
    # Only add reverse pairs where sentences are different
    df_different = df[df["sentence1"] != df["sentence2"]].copy()
    pairs_reverse = df_different[["sentence2", "sentence1"]].copy()
    pairs_reverse.columns = ["query", "positive"]
    
    # Combine both directions
    all_pairs = pd.concat([pairs_forward, pairs_reverse], ignore_index=True)
    n_pairs = len(all_pairs)

    dist.barrier()
    if rank == 0:
        print(f"Pairs building done in {(time.time()-start)/60:.2f} min")
        print(f"Total pairs (with switching): {return_formatted(n_pairs)}")
        start = time.time()
        print("Building corpus and query mappings...")
    
    # Build corpus from all unique sentences
    all_unique_sentences = pd.concat([
        df["sentence1"],
        df["sentence2"]
    ]).drop_duplicates().sort_values().reset_index(drop=True)  # Sort for reproducibility
    
    # Create sentence to doc ID mapping
    text_to_doc_id = pd.Series(
        [f"doc_{i}" for i in range(len(all_unique_sentences))],
        index=all_unique_sentences.values
    )
    
    # Build unique query mapping
    unique_query_mask = ~all_pairs["query"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_texts = all_pairs.loc[unique_query_mask, "query"].tolist()
    unique_query_ids = [f"query_{i}" for i in range(len(unique_query_texts))]
    
    # Create query text to ID mapping
    query_text_to_id = pd.Series(unique_query_ids, index=unique_query_texts)
    
    # Map query and positive texts to IDs
    query_ids = all_pairs["query"].map(query_text_to_id).tolist()
    positive_ids = all_pairs["positive"].map(text_to_doc_id).tolist()
    
    # Get unique positives (in STS, all corpus sentences can be positives)
    unique_positive_ids_series = all_pairs["positive"].map(text_to_doc_id).drop_duplicates()
    unique_positive_ids = unique_positive_ids_series.tolist()
    unique_positive_texts = [all_unique_sentences.iloc[text_to_doc_id[text_to_doc_id == pid].index[0]] 
                             for pid in unique_positive_ids]
    
    # Create ID to text mapping
    id_to_text = {v: k for k, v in text_to_doc_id.items()}
    
    dist.barrier()
    if rank == 0:
        print(f"Corpus and query mapping done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(unique_query_ids))} unique queries before limiting")
        print(f"Found {return_formatted(len(unique_positive_ids))} unique positives")
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
        
        # Use the unified limit_number_of_queries function
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
        
        # Add remaining corpus sentences not in positives
        seen_positive_ids_set = set(unique_positive_ids)
        all_document_ids_set = set(document_ids)
        remaining_corpus_ids = [
            doc_id for doc_id in text_to_doc_id.values
            if doc_id not in seen_positive_ids_set and doc_id not in all_document_ids_set
        ]
        
        if remaining_corpus_ids:
            document_ids.extend(remaining_corpus_ids)
            document_texts.extend([id_to_text[did] for did in remaining_corpus_ids])
        
        if rank == 0:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
            print(f"Positives referenced by filtered pairs: {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")
    else:
        # No limiting needed
        # In STS, typically all documents can be positives, but organize with actual positives first
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [doc_id for doc_id in text_to_doc_id.values if doc_id not in seen_positive_ids_set]
        
        # Build unified document lists: positives first, then remaining
        document_ids = unique_positive_ids + remaining_doc_ids
        document_texts = [id_to_text[doc_id] for doc_id in document_ids]
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(
                f"Found {return_formatted(len(unique_query_ids))} unique queries"
            )
            print(f"Total number of query-positive pairs: {return_formatted(len(query_ids))}")
            print(f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")

    corpus_dict = {
        doc_id: {"text": text} for doc_id, text in zip(document_ids, document_texts)
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
