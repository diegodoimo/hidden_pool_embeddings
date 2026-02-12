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
        print("finding unique items preprocessing")

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
    unique_query_texts = query_texts.iloc[unique_query_idx].reset_index(drop=True)
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]

    if rank == 0:
        print(f"queries done in {(time.time()-start)/60}min")
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

    # Use unique positives as the corpus (documents = positives in this loader)
    documents_are_positives = True

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=None,
        document_ids=None,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=has_title,
        documents_are_positives=documents_are_positives,
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

    # Extract all documents from corpus
    document_ids = list(corpus_dict.keys())
    if has_title:
        document_texts = [entry["text"] for entry in corpus_dict.values()]
        document_titles = [entry["title"] for entry in corpus_dict.values()]
    else:
        document_texts = [entry["text"] for entry in corpus_dict.values()]
        document_titles = None

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        positive_titles=positive_titles,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=has_title,
        documents_are_positives=False,
    )
