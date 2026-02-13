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


def return_formatted(ndata):
    for threshold, suffix in [(10**6, "M"), (10**3, "k")]:
        if ndata >= threshold:
            return f"{ndata / threshold:.3f}{suffix}"
    return str(ndata)


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

    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60}min")
        start = time.time()
        print(
            f"finding unique items preprocessing {return_formatted(len(dataset))} pairs..."
        )

    n_pairs = len(dataset)

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

    # Generate sequential IDs
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]

    if rank == 0:
        print(f"preprocessing done in {(time.time()-start)/60}min")
        start = time.time()
        print("finding unique queries items...")

    # Fast deduplication via pandas C-optimized hash tables.
    unique_query_mask = ~query_texts.duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index

    # Limit to maximum 1 million unique queries
    max_queries = 1_000_000
    if len(unique_query_idx) > max_queries:
        if rank == 0:
            print(f"Limiting queries from {len(unique_query_idx)} to {max_queries}")
        unique_query_idx = unique_query_idx[:max_queries]

    unique_query_texts = query_texts.iloc[unique_query_idx].reset_index(drop=True)
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]

    # Filter query_ids and positive_ids to only include pairs with queries in the limited set
    unique_query_idx_set = set(unique_query_idx)
    valid_pair_mask = [i in unique_query_idx_set for i in range(n_pairs)]
    query_ids = [qid for qid, valid in zip(query_ids, valid_pair_mask) if valid]
    positive_ids = [pid for pid, valid in zip(positive_ids, valid_pair_mask) if valid]

    if rank == 0:
        print(f"queries done in {(time.time()-start)/60}min")
        print(
            f"Kept {len(query_ids)} query-positive pairs after limiting to {len(unique_query_ids)} unique queries"
        )
        start = time.time()
        print("finding unique positives items...")

    unique_positive_mask = ~positive_texts.duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index
    unique_positive_texts = positive_texts.iloc[unique_positive_idx].reset_index(
        drop=True
    )
    unique_positive_ids = [f"doc_{i}" for i in unique_positive_idx]

    if has_title:
        unique_positive_titles = (
            df[title_col].iloc[unique_positive_idx].reset_index(drop=True)
        )
    else:
        unique_positive_titles = None

    del unique_query_mask, unique_positive_mask

    if rank == 0:
        print(
            f"Found {len(unique_query_texts)//10**3}k unique queries out of {n_pairs}"
        )
        print(
            f"Found {len(unique_positive_texts)//10**3}k unique positives out of {n_pairs}"
        )
        print(f"unique items found in {(time.time()-start)/60}min")
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

    if rank == 0:
        print(f"corpus dict built in {(time.time()-start)/60}min")

    # In this function, all documents are positives, so we just use unique positives
    # Documents = unique positives (all positives are at the beginning)
    document_texts = unique_positive_texts
    document_ids = unique_positive_ids
    document_titles = unique_positive_titles
    n_positives = len(unique_positive_ids)

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

    # Limit to maximum 1 million unique queries
    max_queries = 1_000_000
    if len(unique_query_ids) > max_queries:
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

        # Rebuild unique positives based on filtered pairs
        seen_positives_new = set()
        new_unique_positive_ids = []
        new_unique_positive_texts = []
        new_unique_positive_titles = [] if has_title else None

        for pid in positive_ids:
            if pid not in seen_positives_new:
                seen_positives_new.add(pid)
                new_unique_positive_ids.append(pid)
                entry = corpus_dict[pid]
                new_unique_positive_texts.append(entry["text"])
                if has_title:
                    new_unique_positive_titles.append(entry["title"])

        unique_positive_ids = new_unique_positive_ids
        unique_positive_texts = new_unique_positive_texts
        if has_title:
            unique_positive_titles = new_unique_positive_titles

        if rank == 0:
            print(
                f"Kept {len(query_ids)} query-positive pairs after limiting to {len(unique_query_ids)} unique queries"
            )

    # Create unified document lists: unique positives first, then remaining documents
    seen_positive_ids_set = set(unique_positive_ids)
    remaining_doc_ids = [
        doc_id for doc_id in corpus_dict.keys() if doc_id not in seen_positive_ids_set
    ]

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
        task.qrels_fields["score"],
    ]
    df_qrels = qrels.select_columns(qrels_cols).to_pandas()

    # Rename columns for easier access
    df_qrels.columns = ["query_id", "positive_id", "score"]

    # Filter valid pairs: score >= 1, query_id in queries_dict, positive_id in corpus_dict
    valid_queries = df_qrels["query_id"].isin(queries_dict.keys())
    valid_positives = df_qrels["positive_id"].isin(corpus_dict.keys())
    valid_scores = df_qrels["score"] >= 1
    df_qrels = df_qrels[valid_queries & valid_positives & valid_scores].reset_index(
        drop=True
    )

    if rank == 0:
        print(f"Found {len(df_qrels)} valid query-positive pairs")

    # Extract query and positive IDs for all pairs
    query_ids = df_qrels["query_id"].tolist()
    positive_ids = df_qrels["positive_id"].tolist()

    # Get unique queries (preserving first occurrence order)
    unique_query_mask = ~df_qrels["query_id"].duplicated(keep="first")
    unique_query_ids = df_qrels.loc[unique_query_mask, "query_id"].tolist()
    unique_query_texts = [queries_dict[qid]["text"] for qid in unique_query_ids]

    # Limit to maximum 1 million unique queries
    max_queries = 1_000_000
    if len(unique_query_ids) > max_queries:
        if rank == 0:
            print(f"Limiting queries from {len(unique_query_ids)} to {max_queries}")

        # Keep only first max_queries unique queries
        limited_query_ids_set = set(unique_query_ids[:max_queries])
        unique_query_ids = unique_query_ids[:max_queries]
        unique_query_texts = unique_query_texts[:max_queries]

        # Filter dataframe to only include pairs with queries in the limited set
        df_qrels = df_qrels[
            df_qrels["query_id"].isin(limited_query_ids_set)
        ].reset_index(drop=True)

        # Update query_ids and positive_ids from filtered dataframe
        query_ids = df_qrels["query_id"].tolist()
        positive_ids = df_qrels["positive_id"].tolist()

    # Get unique positives (preserving first occurrence order) - recalculate after potential filtering
    unique_positive_mask = ~df_qrels["positive_id"].duplicated(keep="first")
    unique_positive_ids = df_qrels.loc[unique_positive_mask, "positive_id"].tolist()

    if rank == 0:
        print(f"Found {len(unique_query_ids)} unique queries")
        print(f"Found {len(unique_positive_ids)} unique positives")
        print(f"Total query-positive pairs: {len(query_ids)}")

    # Build unified document lists: unique positives first, then remaining corpus docs
    seen_positive_ids_set = set(unique_positive_ids)
    remaining_doc_ids = [
        doc_id for doc_id in corpus_dict.keys() if doc_id not in seen_positive_ids_set
    ]

    # Concatenate: positives first, then remaining
    document_ids = unique_positive_ids + remaining_doc_ids

    if has_title:
        unique_positive_texts = [
            corpus_dict[pid]["text"] for pid in unique_positive_ids
        ]
        unique_positive_titles = [
            corpus_dict[pid]["title"] for pid in unique_positive_ids
        ]
        document_texts = unique_positive_texts + [
            corpus_dict[did]["text"] for did in remaining_doc_ids
        ]
        document_titles = unique_positive_titles + [
            corpus_dict[did]["title"] for did in remaining_doc_ids
        ]
    else:
        unique_positive_texts = [
            corpus_dict[pid]["text"] for pid in unique_positive_ids
        ]
        unique_positive_titles = None
        document_texts = unique_positive_texts + [
            corpus_dict[did]["text"] for did in remaining_doc_ids
        ]
        document_titles = None

    n_positives = len(unique_positive_ids)

    if rank == 0:
        print(
            f"Built unified document list with {n_positives} positives first, {len(remaining_doc_ids)} additional docs"
        )
        print(f"Total unique documents: {len(document_ids)}")
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
