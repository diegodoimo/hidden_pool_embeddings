"""
Shared loader functions for retrieval tasks.
These loaders are used by multiple retrieval tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import time

from tasks.data_helpers import RetrievalRawData, get_dict


def from_one_hf_dataset(task) -> RetrievalRawData:
    """
    Load data from a single HuggingFace dataset where queries and positives
    are in the same dataset with matching indices.
    
    Used by: NaturalQuestions, ALL_NLI, PAQ, ELI5, TriviaQA, COLIEE, 
             S2ORC*, SPECTER, SentenceCompression, StackExchangeDup*, QQP, AmazonQA
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    # Assume dataset has matching lengths and indices correspond to pairs
    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])
    document_texts = list(dataset[task.positive_name])

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]

    # Documents use same IDs as positives
    document_ids = positive_ids.copy()
    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Check if titles exist in dataset
    has_corpus_fields = task.corpus_fields is not None
    has_title = has_corpus_fields and task.corpus_fields.get("title", None) is not None
    if has_title and task.corpus_fields["title"] in dataset.column_names:
        positive_titles = list(dataset[task.corpus_fields["title"]])
        document_titles = positive_titles.copy()
    else:
        has_title = False
        positive_titles = None
        document_titles = None

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=positive_titles,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=query_texts,
        unique_query_ids=query_ids,
        unique_positive_texts=positive_texts,
        unique_positive_ids=positive_ids,
        unique_positive_titles=positive_titles,
        corpus_dict=corpus_dict,
        has_title=has_title,
    )


def from_multiple_hf_datasets(task, rank=0) -> RetrievalRawData:
    """
    Load data from multiple HuggingFace datasets (queries, corpus, qrels).
    This is the standard MTEB format without deduplication.
    
    Used by: MSMARCOv2, BioASQ
    """
    if rank == 0:
        print("Loading datasets...")
    qrels = load_dataset(task.hf_name, name=task.qrels_name, split=task.split)
    anchors_ = load_dataset(task.hf_name, name=task.anchor_name, split=task.anchor_name)
    corpus = load_dataset(
        task.hf_name, name=task.positive_name, split=task.positive_name
    )

    if rank == 0:
        print(f"Mapping {len(anchors_)} queries to dict...")
        start = time.time()
    queries_dict = get_dict(
        anchors_, task.anchor_fields["id"], task.anchor_fields["text"]
    )

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
    if rank == 0:
        print(f"{(time.time() - start)/60: .2f} min for {len(corpus_dict)} samples")
        print("Extracting positives from qrels...")

    query_ids = []
    query_texts = []
    positive_ids = []
    positive_texts = []
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
        query_texts.append(queries_dict[anchor_id]["text"])

        # Extract positive
        positive_entry = corpus_dict[positive_id]
        positive_ids.append(positive_id)

        if has_title:
            positive_texts.append(positive_entry["text"])
            positive_titles.append(positive_entry["title"])
        else:
            positive_texts.append(positive_entry["text"])

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
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
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
    )


def from_multiple_hf_datasets_with_dedup(task, rank=0, eval_split: str = "test") -> RetrievalRawData:
    """
    Load MTEB-style dataset with deduplication against evaluation split.
    Removes any training queries that appear in the evaluation set.
    
    Used by: MSMARCO, NFCorpus, FEVER, HotpotQA, FiQA2018, MrTyDi, SciFact
    """
    if rank == 0:
        print("Loading datasets with deduplication...")
        print(f"Loading eval queries from {task.hf_name} ({eval_split} split)...")

    # Load training data
    qrels = load_dataset(task.hf_name, name=task.qrels_name, split=task.split)
    anchors_ = load_dataset(task.hf_name, name=task.anchor_name, split=task.anchor_name)
    corpus = load_dataset(
        task.hf_name, name=task.positive_name, split=task.positive_name
    )

    if rank == 0:
        print(f"Mapping {len(anchors_)} queries to dict...")
        start = time.time()
    queries_dict = get_dict(
        anchors_, task.anchor_fields["id"], task.anchor_fields["text"]
    )

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
    if rank == 0:
        print(f"{(time.time() - start)/60: .2f} min for {len(corpus_dict)} samples")
        print("Extracting positives from qrels with deduplication...")

    query_ids = []
    query_texts = []
    positive_ids = []
    positive_texts = []
    positive_titles = [] if has_title else None

    unique_query_ids = []
    unique_query_texts = []
    unique_positive_ids = []
    unique_positive_texts = []
    unique_positive_titles = [] if has_title else None

    seen_queries = set()
    seen_positives = set()
    excluded_count = 0

    for qrel in qrels:
        anchor_id = qrel[task.qrels_fields["anchor_id"]]
        positive_id = qrel[task.qrels_fields["positive_id"]]
        score = qrel[task.qrels_fields["score"]]

        # Filter invalid pairs
        if anchor_id not in queries_dict or positive_id not in corpus_dict or score < 1:
            continue

        query_text = queries_dict[anchor_id]["text"]

        # Extract query
        query_ids.append(anchor_id)
        query_texts.append(query_text)

        # Extract positive
        positive_entry = corpus_dict[positive_id]
        positive_ids.append(positive_id)

        if has_title:
            positive_texts.append(positive_entry["text"])
            positive_titles.append(positive_entry["title"])
        else:
            positive_texts.append(positive_entry["text"])

        # queries can be repeated many times in the search
        # for negatives we just want unique queries
        if anchor_id not in seen_queries:
            seen_queries.add(anchor_id)
            unique_query_ids.append(anchor_id)
            unique_query_texts.append(query_text)

        # positives can also be repeated, select them independently
        if positive_id not in seen_positives:
            seen_positives.add(positive_id)
            unique_positive_ids.append(positive_id)

            if has_title:
                unique_positive_texts.append(positive_entry["text"])
                unique_positive_titles.append(positive_entry["title"])
            else:
                unique_positive_texts.append(positive_entry["text"])

    if rank == 0:
        print(f"Excluded {excluded_count} query-doc pairs due to eval overlap")

    # Extract all documents from corpus
    document_ids = list(corpus_dict.keys())
    if has_title:
        document_texts = [entry["text"] for entry in corpus_dict.values()]
        document_titles = [entry["title"] for entry in corpus_dict.values()]
    else:
        document_texts = [entry["text"] for entry in corpus_dict.values()]
        document_titles = None

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
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
    )
