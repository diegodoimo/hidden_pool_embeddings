"""
Shared loader functions for retrieval tasks.
These loaders are used by multiple retrieval tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import time
import torch.distributed as dist
import pandas as pd
from tasks.data_helpers import RetrievalRawData, get_dict
from utils.helpers import return_formatted


def from_one_hf_dataset(task) -> RetrievalRawData:
    """
    Load data from a single HuggingFace dataset where queries and positives
    are in the same dataset with matching indices.

    Used by: NaturalQuestions, ALL_NLI, PAQ, ELI5, TriviaQA, COLIEE,
             S2ORC*, SPECTER, SentenceCompression, StackExchangeDup*, QQP, AmazonQA
    """
    rank = dist.get_rank()

    if rank == 0:
        start = time.time()
        print("Loading datasets...")
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    n_pairs = len(dataset)

    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60}min")
        start = time.time()
        print(f"num elements in dataset: {return_formatted(n_pairs)}")
        print("building dataframes")

    # Check if titles exist in dataset
    has_corpus_fields = task.corpus_fields is not None
    has_title = has_corpus_fields and task.corpus_fields.get("title", None) is not None
    title_col = None
    if has_title:
        title_col = task.corpus_fields.get("title", None)
        if title_col not in dataset.column_names:
            has_title = False
            title_col = None

    # Convert Arrow -> pandas DataFrame in one shot (fast columnar conversion),
    # avoiding the slow path of dataset[col] (Python list) -> pd.Series.
    cols_to_load = [task.anchor_name, task.positive_name]
    if has_title:
        cols_to_load.append(title_col)
    df = dataset.select_columns(cols_to_load).to_pandas()

    # Keep as pandas Series — no .tolist() needed.
    # Dataset.from_dict() in dict_to_dataset() accepts Series directly,
    # so the round-trip Arrow → list → Arrow is avoided for 20M strings.
    query_texts = df[task.anchor_name]
    positive_texts = df[task.positive_name]

    dist.barrier()
    if rank == 0:
        print(f"preprocessing done in {(time.time()-start)/60}min")
        start = time.time()
        print("finding unique queries items...")

    # Fast deduplication via pandas C-optimized hash tables.
    unique_query_mask = ~query_texts.duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index
    unique_query_texts = query_texts.iloc[unique_query_idx]
    # Use first occurrence indices as IDs
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]

    if rank == 0:
        print(f"queries done in {(time.time()-start)/60}min")
        start = time.time()
        print("finding unique positives items...")

    unique_positive_mask = ~positive_texts.duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index
    unique_positive_texts = positive_texts.iloc[unique_positive_idx]
    # Use first occurrence indices as IDs
    unique_positive_ids = [f"doc_{i}" for i in unique_positive_idx]
    n_positives = len(unique_positive_ids)

    if has_title:
        unique_positive_titles = df[title_col].iloc[unique_positive_idx]
    else:
        unique_positive_titles = None

    if rank == 0:
        print(f"positives done in {(time.time()-start)/60}min")
        start = time.time()
        print("remapping original indices based on unique query positive indices...")


    # Map each text to its first occurrence index for ID remapping
    # This ensures duplicates point to the same ID as their first occurrence
    query_text_to_first_idx = {}
    for idx in unique_query_idx:
        text = query_texts.iloc[idx]
        query_text_to_first_idx[text] = idx
    
    positive_text_to_first_idx = {}
    for idx in unique_positive_idx:
        text = positive_texts.iloc[idx]
        positive_text_to_first_idx[text] = idx
    
    # Generate remapped IDs: map all occurrences (including duplicates) to first occurrence ID
    query_ids = [f"query_{query_text_to_first_idx[query_texts.iloc[i]]}" for i in range(n_pairs)]
    referenced_positive_ids = [f"doc_{positive_text_to_first_idx[positive_texts.iloc[i]]}" for i in range(n_pairs)]

    del unique_query_mask, unique_positive_mask

    dist.barrier()
    if rank == 0:
        print(f"remapping done in {(time.time()-start)/60}min")
        start = time.time()
        print("generating corpus dict...")

    # Since documents = positives in this function, use unique positives for corpus
    # This ensures corpus_dict has unique entries (bijective doc_id <-> document)
    if has_title:
        corpus_dict = {
            id_: {"text": doc_text, "title": doc_title}
            for id_, doc_text, doc_title in zip(
                unique_positive_ids, unique_positive_texts, unique_positive_titles
            )
        }
    else:
        corpus_dict = {
            id_: {"text": doc_text}
            for id_, doc_text in zip(unique_positive_ids, unique_positive_texts)
        }

    dist.barrier()
    if rank == 0:
        print(f"corpus dict built in {(time.time()-start)/60}min")


    assert set(referenced_positive_ids).issubset(
            set(unique_positive_ids)
        ), "filtered qrels contain positive IDs not in corpus"

    # Apply query limiting only if needed
    max_queries = 10**6
    if len(unique_query_idx) > max_queries:
        if rank == 0:
            start = time.time()
            print(f"number of unique_queries {return_formatted(len(unique_query_idx))} > 1M: removing queries")
        # Apply query limiting and reorganize documents
        (
            unique_query_texts,
            unique_query_ids,
            query_ids,
            referenced_positive_ids,
            unique_positive_ids,
            unique_positive_texts,
            unique_positive_titles,
            n_positives
        ) = limit_number_of_queries(
            unique_query_idx=unique_query_idx,
            query_texts=query_texts,
            n_pairs=n_pairs,
            query_ids=query_ids,
            positive_ids=referenced_positive_ids,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            has_title=has_title,
            max_queries=max_queries
        )
        if rank == 0:
            print(f"queries pruned {(time.time()-start)/60}min")
            print(f"Total unique documents in corpus: {return_formatted(len(unique_positive_ids))}")
            print(f"Positives referenced by filtered pairs: {return_formatted(len(referenced_positive_ids))}")       
    
    dist.barrier()
    if rank == 0:
        print(f"Found {return_formatted(len(unique_query_texts))} unique queries (under {max_queries//10**6}M limit)")
        print(f"Total unique documents in corpus: {return_formatted(len(unique_positive_ids))}")
    

    assert set(referenced_positive_ids).issubset(
            set(unique_positive_ids)
        ), "filtered qrels contain positive IDs not in corpus"

    assert set(unique_positive_ids) == set(corpus_dict.keys())

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=referenced_positive_ids,
        document_texts=unique_positive_texts,
        document_ids=unique_positive_ids,
        document_titles=unique_positive_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=has_title,
        n_positives=n_positives,
    )


def limit_number_of_queries(
    unique_query_idx,
    query_texts,
    n_pairs,
    query_ids,
    positive_ids,
    unique_positive_ids,
    unique_positive_texts,
    unique_positive_titles,
    has_title,
    max_queries=10**6
):
    """
    Limit the number of queries to max_queries and adjust all related data structures.
    
    Args:
        unique_query_idx: Index of unique queries in original data
        query_texts: Series of all query texts
        n_pairs: Total number of query-positive pairs
        query_ids: List of query IDs for each pair
        positive_ids: List of positive IDs for each pair
        unique_positive_ids: List of all unique positive IDs
        unique_positive_texts: List of all unique positive texts
        unique_positive_titles: List of all unique positive titles (or None)
        has_title: Boolean indicating if titles exist
        rank: Distributed training rank
        max_queries: Maximum number of queries to keep (default: 1 million)
    
    Returns:
        Tuple of: (unique_query_texts, unique_query_ids, query_ids, positive_ids,
                   document_ids, document_texts, document_titles, n_positives)
    """
    
    unique_query_idx = unique_query_idx[:max_queries]
    unique_query_texts = query_texts.iloc[unique_query_idx].reset_index(drop=True)
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]
    
    # Filter query_ids and positive_ids to only include pairs with queries in the limited set
    unique_query_idx_set = set(unique_query_idx)
    valid_pair_mask = [i in unique_query_idx_set for i in range(n_pairs)]
    query_ids = [qid for qid, valid in zip(query_ids, valid_pair_mask) if valid]
    positive_ids = [pid for pid, valid in zip(positive_ids, valid_pair_mask) if valid]

    # Reorganize documents: referenced positives first, then unreferenced ones
    referenced_positive_ids_set = set(positive_ids)
    
    # Split into referenced and unreferenced positives
    referenced_pos_ids = []
    referenced_pos_texts = []
    referenced_pos_titles = [] if has_title else None
    
    unreferenced_pos_ids = []
    unreferenced_pos_texts = []
    unreferenced_pos_titles = [] if has_title else None
    
    for i, pos_id in enumerate(unique_positive_ids):
        if pos_id in referenced_positive_ids_set:
            referenced_pos_ids.append(pos_id)
            referenced_pos_texts.append(unique_positive_texts[i])
            if has_title:
                referenced_pos_titles.append(unique_positive_titles[i])
        else:
            unreferenced_pos_ids.append(pos_id)
            unreferenced_pos_texts.append(unique_positive_texts[i])
            if has_title:
                unreferenced_pos_titles.append(unique_positive_titles[i])
    
    # Concatenate: referenced positives first, then unreferenced
    document_ids = referenced_pos_ids + unreferenced_pos_ids
    document_texts = referenced_pos_texts + unreferenced_pos_texts
    if has_title:
        document_titles = referenced_pos_titles + unreferenced_pos_titles
    else:
        document_titles = None
    
    n_positives = len(referenced_pos_ids)
    
    return (
        unique_query_texts,
        unique_query_ids,
        query_ids,
        positive_ids,
        document_ids,
        document_texts,
        document_titles,
        n_positives
    )


def limit_number_of_queries_multi_dataset(
    unique_query_ids,
    unique_query_texts,
    query_ids,
    positive_ids,
    positive_titles,
    unique_positive_ids,
    unique_positive_texts,
    unique_positive_titles,
    corpus_dict,
    has_title,
    rank=0,
    max_queries=10**6
):
    """
    Limit the number of queries to max_queries for multi-dataset loaders and adjust all related data structures.
    
    This version works with list-based data structures (not pandas) and handles corpus_dict.
    
    Args:
        unique_query_ids: List of unique query IDs
        unique_query_texts: List of unique query texts
        query_ids: List of query IDs for each pair
        positive_ids: List of positive IDs for each pair
        positive_titles: List of positive titles for each pair (or None)
        unique_positive_ids: List of unique positive IDs from qrels
        unique_positive_texts: List of unique positive texts from qrels
        unique_positive_titles: List of unique positive titles from qrels (or None)
        corpus_dict: Dictionary mapping doc_id -> {text, title}
        has_title: Boolean indicating if titles exist
        rank: Distributed training rank
        max_queries: Maximum number of queries to keep (default: 1 million)
    
    Returns:
        Tuple of: (unique_query_ids, unique_query_texts, query_ids, positive_ids,
                   positive_titles, document_ids, document_texts, document_titles, n_positives)
    """
    # Limit to maximum queries (function is only called when len > max_queries)
    if rank == 0:
        print(f"Limiting queries from {len(unique_query_ids)} to {max_queries}")
    
    # Keep only first max_queries unique queries
    limited_query_ids_set = set(unique_query_ids[:max_queries])
    unique_query_ids = unique_query_ids[:max_queries]
    unique_query_texts = unique_query_texts[:max_queries]
    
    # Filter query_ids and positive_ids to only include pairs with queries in the limited set
    filtered_query_ids = []
    filtered_positive_ids = []
    filtered_positive_titles = [] if has_title else None
    
    for qid, pid in zip(query_ids, positive_ids):
        if qid in limited_query_ids_set:
            filtered_query_ids.append(qid)
            filtered_positive_ids.append(pid)
    
    if has_title:
        for qid, title in zip(query_ids, positive_titles):
            if qid in limited_query_ids_set:
                filtered_positive_titles.append(title)
    
    query_ids = filtered_query_ids
    positive_ids = filtered_positive_ids
    positive_titles = filtered_positive_titles
    
    if rank == 0:
        print(f"Kept {len(query_ids)} query-positive pairs after limiting to {len(unique_query_ids)} unique queries")
    
    # Reorganize positives: referenced ones first, then unreferenced from qrels
    referenced_positive_ids_set = set(positive_ids)
    
    # Split unique_positive_ids (from qrels) into referenced and unreferenced
    referenced_pos_ids = []
    referenced_pos_texts = []
    referenced_pos_titles = [] if has_title else None
    
    unreferenced_pos_ids = []
    unreferenced_pos_texts = []
    unreferenced_pos_titles = [] if has_title else None
    
    for i, pos_id in enumerate(unique_positive_ids):
        if pos_id in referenced_positive_ids_set:
            referenced_pos_ids.append(pos_id)
            referenced_pos_texts.append(unique_positive_texts[i])
            if has_title:
                referenced_pos_titles.append(unique_positive_titles[i])
        else:
            unreferenced_pos_ids.append(pos_id)
            unreferenced_pos_texts.append(unique_positive_texts[i])
            if has_title:
                unreferenced_pos_titles.append(unique_positive_titles[i])
    
    # Get remaining documents from corpus that weren't in qrels at all
    all_qrels_positive_ids_set = set(unique_positive_ids)
    remaining_doc_ids = [doc_id for doc_id in corpus_dict.keys() if doc_id not in all_qrels_positive_ids_set]
    
    # Build unified document list: referenced positives + unreferenced positives + remaining corpus
    document_ids = referenced_pos_ids + unreferenced_pos_ids + remaining_doc_ids
    document_texts = referenced_pos_texts + unreferenced_pos_texts
    
    if has_title:
        document_titles = referenced_pos_titles + unreferenced_pos_titles
        # Add remaining documents from corpus
        for doc_id in remaining_doc_ids:
            entry = corpus_dict[doc_id]
            document_texts.append(entry["text"])
            document_titles.append(entry["title"])
    else:
        document_titles = None
        # Add remaining documents from corpus
        for doc_id in remaining_doc_ids:
            document_texts.append(corpus_dict[doc_id]["text"])
    
    n_positives = len(referenced_pos_ids)
    
    if rank == 0:
        print(f"Positives referenced by filtered pairs (n_positives): {n_positives}")
        print(f"Unreferenced positives from qrels: {len(unreferenced_pos_ids)}")
        print(f"Additional documents from corpus: {len(remaining_doc_ids)}")
        print(f"Total unique documents in corpus: {len(document_ids)}")
    
    return (
        unique_query_ids,
        unique_query_texts,
        query_ids,
        positive_ids,
        positive_titles,
        document_ids,
        document_texts,
        document_titles,
        n_positives
    )


def from_multiple_hf_datasets(task, rank=0) -> RetrievalRawData:
    """
    Load data from multiple HuggingFace datasets (queries, corpus, qrels).
    This is the standard MTEB format.
    """
    if rank == 0:
        print("Loading datasets...")
    qrels = load_dataset(task.hf_name, name=task.qrels_name, split=task.split)
    anchors_ = load_dataset(task.hf_name, name=task.anchor_name, split=task.anchor_name)
    corpus = load_dataset(
        task.hf_name, name=task.positive_name, split=task.positive_name
    )

    dist.barrier()
    if rank == 0:
        print(f"Mapping {len(anchors_)} queries to dict...")
        start = time.time()
    queries_dict = get_dict(
        anchors_, task.anchor_fields["id"], task.anchor_fields["text"]
    )

    dist.barrier()
    if rank == 0:
        print(f"{(time.time() - start): .2f} sec for {len(queries_dict)} samples")
        start = time.time()
        print(f"Mapping {len(corpus)} docs to dict...")

    corpus_dict = get_dict(
        corpus,
        task.corpus_fields["id"],
        task.corpus_fields["text"],
        task.corpus_fields.get("title", None),
    )

    has_title = task.corpus_fields.get("title", None) is not None
    dist.barrier()
    if rank == 0:
        print(f"{(time.time() - start)/60: .2f} min for {len(corpus_dict)} samples")
        print("Extracting positives from qrels...")

    query_ids = []
    positive_ids = []
    positive_titles = [] if has_title else None

    unique_query_ids = []
    unique_query_texts = []
    unique_positive_ids = []
    unique_positive_texts = []
    unique_positive_titles = [] if has_title else None

    seen_queries = set()
    seen_positives = set()

    for qrel in qrels:
        anchor_id = qrel[task.qrels_fields["anchor_id"]]
        positive_id = qrel[task.qrels_fields["positive_id"]]
        score = qrel[task.qrels_fields["score"]]

        # Filter invalid pairs
        if anchor_id not in queries_dict or positive_id not in corpus_dict or score < 1:
            continue

        # Extract query
        query_ids.append(anchor_id)

        # Extract positive
        positive_entry = corpus_dict[positive_id]
        positive_ids.append(positive_id)

        if has_title:
            positive_titles.append(positive_entry["title"])

        # queries can be repeated many times in the search
        # for negatives we just want unique queries
        if anchor_id not in seen_queries:
            seen_queries.add(anchor_id)
            unique_query_ids.append(anchor_id)
            unique_query_texts.append(queries_dict[anchor_id]["text"])

        # positives can also be repeated, select them independently
        if positive_id not in seen_positives:
            seen_positives.add(positive_id)
            unique_positive_ids.append(positive_id)

            if has_title:
                unique_positive_texts.append(positive_entry["text"])
                unique_positive_titles.append(positive_entry["title"])
            else:
                unique_positive_texts.append(positive_entry["text"])

    # Apply query limiting only if needed
    max_queries = 10**6
    if len(unique_query_ids) > max_queries:
        # Apply query limiting and reorganize documents
        (
            unique_query_ids,
            unique_query_texts,
            query_ids,
            positive_ids,
            positive_titles,
            document_ids,
            document_texts,
            document_titles,
            n_positives
        ) = limit_number_of_queries_multi_dataset(
            unique_query_ids=unique_query_ids,
            unique_query_texts=unique_query_texts,
            query_ids=query_ids,
            positive_ids=positive_ids,
            positive_titles=positive_titles,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            corpus_dict=corpus_dict,
            has_title=has_title,
            rank=rank,
            max_queries=max_queries
        )
    else:
        # No limiting needed, keep data as is
        # Create unified document lists: unique positives first, then remaining documents
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [doc_id for doc_id in corpus_dict.keys() if doc_id not in seen_positive_ids_set]
        
        # Build unified document lists with positives first
        document_ids = unique_positive_ids + remaining_doc_ids
        document_texts = unique_positive_texts.copy()
        
        if has_title:
            document_titles = unique_positive_titles.copy()
            # Add remaining documents
            for doc_id in remaining_doc_ids:
                entry = corpus_dict[doc_id]
                document_texts.append(entry["text"])
                document_titles.append(entry["title"])
        else:
            document_titles = None
            # Add remaining documents
            for doc_id in remaining_doc_ids:
                document_texts.append(corpus_dict[doc_id]["text"])
        
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(f"Found {len(unique_query_ids)//10**3}k unique queries (under {max_queries//10**6}M limit)")
            print(f"Total unique documents in corpus: {len(document_ids)}")

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=has_title,
        n_positives=n_positives,
    )


def from_multiple_hf_datasets_vectorized(task, rank=0) -> RetrievalRawData:
    """
    Load data from multiple HuggingFace datasets (queries, corpus, qrels) using vectorized operations.
    This is the standard MTEB format with pandas-based vectorization for better performance.
    
    Uses the same unified format as from_one_hf_dataset:
    - Documents are unique with positives first
    - n_positives indicates how many positives are at the beginning
    """
    if rank == 0:
        start = time.time()
        print("Loading datasets...")
    
    qrels = load_dataset(task.hf_name, name=task.qrels_name, split=task.split)
    anchors_ = load_dataset(task.hf_name, name=task.anchor_name, split=task.anchor_name)
    corpus = load_dataset(
        task.hf_name, name=task.positive_name, split=task.positive_name
    )

    dist.barrier()
    if rank == 0:
        print(f"Datasets loaded in {(time.time()-start)/60}min")
        start = time.time()
        print(f"num elements in queries: {len(queries)//10**3}k")
        print(f"num elements in qrels: {len(qrels)//10**3}k")
        print(f"num elements in corpus: {len(corpus)//10**3}k")
        print(f"Processing {len(anchors_)} queries...")
    
    # Build queries dict
    queries_dict = get_dict(
        anchors_, task.anchor_fields["id"], task.anchor_fields["text"]
    )

    dist.barrier()
    if rank == 0:
        print(f"Queries processed in {(time.time()-start)/60}min")
        start = time.time()
        print(f"Processing {len(corpus)} docs...")

    # Build corpus dict
    has_title = task.corpus_fields.get("title", None) is not None
    corpus_dict = get_dict(
        corpus,
        task.corpus_fields["id"],
        task.corpus_fields["text"],
        task.corpus_fields.get("title", None),
    )

    dist.barrier()
    if rank == 0:
        print(f"Corpus processed in {(time.time()-start)/60}min")
        start = time.time()
        print("Processing qrels with vectorized operations...")

    # Convert qrels to pandas DataFrame for vectorized operations
    qrels_cols = [
        task.qrels_fields["anchor_id"],
        task.qrels_fields["positive_id"],
        task.qrels_fields["score"]
    ]
    df_qrels = qrels.select_columns(qrels_cols).to_pandas()
    
    # Rename columns for easier access
    df_qrels.columns = ["query_id", "positive_id", "score"]
    
    # Filter valid pairs: score >= 1, query_id in queries_dict, positive_id in corpus_dict
    valid_queries = df_qrels["query_id"].isin(queries_dict.keys())
    valid_positives = df_qrels["positive_id"].isin(corpus_dict.keys())
    valid_scores = df_qrels["score"] >= 1
    df_qrels = df_qrels[valid_queries & valid_positives & valid_scores].reset_index(drop=True)
    
    if rank == 0:
        print(f"Found {len(df_qrels)} valid query-positive pairs")
    
    # Extract query and positive IDs for all pairs
    query_ids = df_qrels["query_id"].tolist()
    positive_ids = df_qrels["positive_id"].tolist()
    
    # Get unique queries (preserving first occurrence order)
    unique_query_mask = ~df_qrels["query_id"].duplicated(keep="first")
    unique_query_ids = df_qrels.loc[unique_query_mask, "query_id"].tolist()
    unique_query_texts = [queries_dict[qid]["text"] for qid in unique_query_ids]
    
    # Get unique positives (preserving first occurrence order) - calculate BEFORE filtering
    unique_positive_mask = ~df_qrels["positive_id"].duplicated(keep="first")
    unique_positive_ids = df_qrels.loc[unique_positive_mask, "positive_id"].tolist()
    
    # Extract texts and titles for unique positives from corpus_dict
    if has_title:
        unique_positive_texts = [corpus_dict[pid]["text"] for pid in unique_positive_ids]
        unique_positive_titles = [corpus_dict[pid]["title"] for pid in unique_positive_ids]
    else:
        unique_positive_texts = [corpus_dict[pid]["text"] for pid in unique_positive_ids]
        unique_positive_titles = None
    
    # Apply query limiting only if needed
    max_queries = 10**6
    if len(unique_query_ids) > max_queries:
        # Apply query limiting and reorganize documents
        (
            unique_query_ids,
            unique_query_texts,
            query_ids,
            positive_ids,
            _,  
            document_ids,
            document_texts,
            document_titles,
            n_positives
        ) = limit_number_of_queries_multi_dataset(
            unique_query_ids=unique_query_ids,
            unique_query_texts=unique_query_texts,
            query_ids=query_ids,
            positive_ids=positive_ids,
            positive_titles=None,  # Not tracked in vectorized version
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            corpus_dict=corpus_dict,
            has_title=has_title,
            rank=rank,
            max_queries=max_queries
        )
    else:
        # No limiting needed, keep data as is
        # Build unified document lists: unique positives first, then remaining corpus docs
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [doc_id for doc_id in corpus_dict.keys() if doc_id not in seen_positive_ids_set]
        
        # Concatenate: positives first, then remaining
        document_ids = unique_positive_ids + remaining_doc_ids
        document_texts = unique_positive_texts + [corpus_dict[did]["text"] for did in remaining_doc_ids]
        
        if has_title:
            document_titles = unique_positive_titles + [corpus_dict[did]["title"] for did in remaining_doc_ids]
        else:
            document_titles = None
        
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(f"Found {len(unique_query_ids)//10**3}k unique queries (under {max_queries//10**6}M limit)")
            print(f"Total unique documents: {len(document_ids)}")
    
    if rank == 0:
        print(f"Qrels processing completed in {(time.time()-start)/60}min")

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=has_title,
        n_positives=n_positives,
    )
