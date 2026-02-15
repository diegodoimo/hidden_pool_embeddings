"""
NLI-specific loader functions.
These loaders handle Natural Language Inference tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import random
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from collections import defaultdict

from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


def load_nli_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """
    Load NLI datasets (SNLI, MNLI, ANLI) for retrieval with hard negatives using vectorized operations.

    For natural language inference (NLI) datasets, we retain only premises with at least one
    entailed hypothesis as queries, and sample one of the entailed hypotheses as the positive.
    If the query is also paired with neutral or contradictory hypotheses, we add them to the
    corpus so they can be mined as hard negatives.

    Used by: SNLI, MNLI, ANLI, XNLI
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset...")
    
    if hasattr(task, "hf_subset") and task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)
    
    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(len(dataset))}")
        start = time.time()
        print("Converting to pandas and processing...")

    # Get label configuration
    entailment_label = getattr(task, "entailment_label", 0)
    neutral_label = getattr(task, "neutral_label", 1)
    contradiction_label = getattr(task, "contradiction_label", 2)

    # Convert to pandas DataFrame for vectorized operations
    cols_to_load = [task.anchor_name, task.positive_name, task.label_name]
    df = dataset.select_columns(cols_to_load).to_pandas()
    df.columns = ["premise", "hypothesis", "label"]
    
    # Filter out invalid labels (e.g., -1 in SNLI)
    df = df[df["label"] >= 0].reset_index(drop=True)
    
    dist.barrier()
    if rank == 0:
        print(f"Conversion done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Grouping by premise...")
    
    # Create label categories
    df["is_entailment"] = df["label"] == entailment_label
    df["is_non_entailment"] = df["label"].isin([neutral_label, contradiction_label])
    
    # Group by premise to find premises with at least one entailment
    premise_groups = df.groupby("premise").agg({
        "is_entailment": "sum",
        "is_non_entailment": "sum"
    })
    
    # Keep only premises with at least one entailment
    valid_premises = premise_groups[premise_groups["is_entailment"] > 0].index.tolist()
    df_valid = df[df["premise"].isin(valid_premises)].reset_index(drop=True)
    
    dist.barrier()
    if rank == 0:
        print(f"Grouping done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(valid_premises))} valid premises with entailment")
        start = time.time()
        print("Building query-positive pairs...")

    # Build unique query ID mapping (one-to-one: unique premise -> unique query ID)
    unique_premises = pd.Series(valid_premises).drop_duplicates()
    unique_query_ids = [f"query_{i}" for i in range(len(unique_premises))]
    unique_query_texts = unique_premises.tolist()
    premise_to_query_id = pd.Series(unique_query_ids, index=unique_premises.values)
    
    # For each premise, sample one entailed hypothesis as positive
    # Group entailments by premise
    df_entailments = df_valid[df_valid["is_entailment"]].copy()
    
    # Sample one entailment per premise (using groupby + sample)
    sampled_positives = df_entailments.groupby("premise")["hypothesis"].apply(
        lambda x: x.sample(n=1, random_state=42).iloc[0]
    ).reset_index()
    sampled_positives.columns = ["premise", "positive_hypothesis"]
    
    # Map premises to query IDs
    sampled_positives["query_id"] = sampled_positives["premise"].map(premise_to_query_id)
    
    # Build unique hypotheses (all hypotheses, both entailment and non-entailment)
    all_hypotheses = df_valid["hypothesis"].drop_duplicates().reset_index(drop=True)
    hypothesis_to_id = pd.Series(
        [f"doc_{i}" for i in range(len(all_hypotheses))],
        index=all_hypotheses.values
    )
    
    # Map positive hypotheses to doc IDs
    sampled_positives["positive_id"] = sampled_positives["positive_hypothesis"].map(hypothesis_to_id)
    
    # Build query-positive pairs
    query_ids = sampled_positives["query_id"].tolist()
    positive_ids = sampled_positives["positive_id"].tolist()
    n_pairs = len(query_ids)

    dist.barrier()
    if rank == 0:
        print(f"Query-positive pairs built in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Building corpus and applying query limiting...")
    
    # Get unique positives
    unique_positive_ids_series = pd.Series(positive_ids).drop_duplicates()
    unique_positive_ids = unique_positive_ids_series.tolist()
    unique_positive_texts = [all_hypotheses.iloc[hypothesis_to_id[hypothesis_to_id == pid].index[0]] 
                             for pid in unique_positive_ids]
    
    # Create ID to hypothesis mapping
    id_to_hypothesis = {v: k for k, v in hypothesis_to_id.items()}
    
    # Apply query limiting if needed
    unique_query_idx = np.arange(len(unique_query_ids))
    
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
        
        # Add remaining corpus hypotheses not in positives
        seen_positive_ids_set = set(unique_positive_ids)
        all_document_ids_set = set(document_ids)
        remaining_corpus_ids = [
            doc_id for doc_id in hypothesis_to_id.values()
            if doc_id not in seen_positive_ids_set and doc_id not in all_document_ids_set
        ]
        
        if remaining_corpus_ids:
            document_ids.extend(remaining_corpus_ids)
            document_texts.extend([id_to_hypothesis[did] for did in remaining_corpus_ids])
        
        if rank == 0:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
            print(f"Positives referenced by filtered pairs: {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")
    else:
        # No limiting needed
        # Collect remaining documents (non-positives)
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [
            doc_id for doc_id in hypothesis_to_id.values()
            if doc_id not in seen_positive_ids_set
        ]
        
        # Build unified document lists: positives first, then remaining
        document_ids = unique_positive_ids + remaining_doc_ids
        document_texts = [id_to_hypothesis[doc_id] for doc_id in document_ids]
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(
                f"Found {return_formatted(len(unique_query_ids))} unique queries"
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


def load_all_nli_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """
    Load ALL_NLI dataset which already has triplets (anchor, positive, negative) using vectorized operations.
    The negatives are included in the corpus so they can be mined as hard negatives.

    Used by: ALL_NLI
    
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
        print("Converting to pandas...")

    # Check if negative field exists
    has_negatives = (
        hasattr(task, "negative_name") and task.negative_name in dataset.column_names
    )
    
    # Convert to pandas DataFrame
    cols_to_load = [task.anchor_name, task.positive_name]
    if has_negatives:
        cols_to_load.append(task.negative_name)
    
    df = dataset.select_columns(cols_to_load).to_pandas()
    
    if has_negatives:
        df.columns = ["query_text", "positive_text", "negative_text"]
    else:
        df.columns = ["query_text", "positive_text"]
    
    n_pairs = len(df)
    
    dist.barrier()
    if rank == 0:
        print(f"Conversion done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Building unique queries and corpus...")
    
    # Build unique query mapping using pandas
    unique_query_mask = ~df["query_text"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_texts = df.loc[unique_query_mask, "query_text"].tolist()
    unique_query_ids = [f"query_{i}" for i in range(len(unique_query_texts))]
    
    # Create query text to ID mapping
    query_text_to_id = pd.Series(unique_query_ids, index=unique_query_texts)
    
    # Build corpus from all unique texts (positives + negatives)
    if has_negatives:
        all_corpus_texts = pd.concat([
            df["positive_text"],
            df["negative_text"]
        ]).drop_duplicates().reset_index(drop=True)
    else:
        all_corpus_texts = df["positive_text"].drop_duplicates().reset_index(drop=True)
    
    # Create text to doc ID mapping
    text_to_id = pd.Series(
        [f"doc_{i}" for i in range(len(all_corpus_texts))],
        index=all_corpus_texts.values
    )
    
    # Map query texts and positive texts to IDs
    query_ids = df["query_text"].map(query_text_to_id).tolist()
    positive_ids = df["positive_text"].map(text_to_id).tolist()

    # Get unique positives (only from actual positives, not negatives)
    unique_positive_mask = ~df["positive_text"].duplicated(keep="first")
    unique_positive_ids_series = df.loc[unique_positive_mask, "positive_text"].map(text_to_id)
    unique_positive_ids = unique_positive_ids_series.tolist()
    unique_positive_texts = df.loc[unique_positive_mask, "positive_text"].tolist()
    
    # Create ID to text mapping for corpus reconstruction
    id_to_text = {v: k for k, v in text_to_id.items()}
    
    dist.barrier()
    if rank == 0:
        print(f"Corpus building done in {(time.time()-start)/60:.2f} min")
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
        
        # Add remaining corpus texts not in positives
        seen_positive_ids_set = set(unique_positive_ids)
        all_document_ids_set = set(document_ids)
        remaining_corpus_ids = [
            doc_id for doc_id in text_to_id.values()
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
        # Build corpus with positives first, then remaining documents
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [doc_id for doc_id in text_to_id.values() if doc_id not in seen_positive_ids_set]
        
        # Unified document lists: positives first
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
