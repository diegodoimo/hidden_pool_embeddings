from datasets import load_dataset

from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value
import time
import os
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
from .data_helpers import RetrievalRawData, get_dict


def normalize_text(
    text: str,
) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def extract_unique_queries(
    query_texts: List[str],
    query_ids: List[str],
    positive_texts: List[str],
    positive_ids: List[str],
    positive_titles: Optional[List[str]] = None,
) -> tuple:
    """
    Extract unique queries from lists that may contain repeated queries.
    
    Returns:
        tuple of (unique_query_texts, unique_query_ids, unique_positive_texts, 
                  unique_positive_ids, unique_positive_titles)
    """
    seen_query_texts = set()
    unique_query_texts = []
    unique_query_ids = []
    unique_positive_texts = []
    unique_positive_ids = []
    unique_positive_titles = [] if positive_titles is not None else None
    
    for i, query_text in enumerate(query_texts):
        if query_text not in seen_query_texts:
            seen_query_texts.add(query_text)
            unique_query_texts.append(query_text)
            unique_query_ids.append(query_ids[i])
            unique_positive_texts.append(positive_texts[i])
            unique_positive_ids.append(positive_ids[i])
            if positive_titles is not None:
                unique_positive_titles.append(positive_titles[i])
    
    return (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    )


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

    Args:
        dataset: The BeIR/arguana dataset
        anchor_field: Field name for queries (e.g., 'query')
        positive_field: Field name for positives (e.g., 'text')
        mteb_query_texts: Set of normalized query texts from mteb/arguana
        mteb_corpus_texts: Set of normalized corpus texts from mteb/arguana

    Returns:
        Filtered dataset with overlapping examples removed
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


def get_mteb_eval_queries(hf_name: str, eval_split: str = "test") -> Set[str]:
    """
    Load MTEB evaluation queries and return a set of normalized query texts.
    
    Args:
        hf_name: HuggingFace dataset name (e.g., 'mteb/nfcorpus')
        eval_split: The evaluation split to load queries from (default: 'test')
    
    Returns:
        Set of normalized query texts from the evaluation split
    """
    queries = load_dataset(hf_name, name="queries", split="queries")
    qrels = load_dataset(hf_name, name="default", split=eval_split)
    
    # Get query IDs that appear in eval split
    eval_query_ids = {qrel["query-id"] for qrel in qrels}
    
    # Build set of normalized query texts for eval queries
    eval_query_texts = set()
    for row in queries:
        if row["_id"] in eval_query_ids:
            eval_query_texts.add(normalize_text(row["text"]))
    
    return eval_query_texts


def filter_train_queries_against_eval(
    train_query_ids: List[str],
    train_query_texts: List[str],
    eval_query_texts: Set[str],
) -> Set[str]:
    """
    Find train query IDs that overlap with eval query texts.
    
    Returns:
        Set of train query IDs to exclude
    """
    exclude_ids = set()
    for qid, qtext in zip(train_query_ids, train_query_texts):
        if normalize_text(qtext) in eval_query_texts:
            exclude_ids.add(qid)
    return exclude_ids


def from_multiple_hf_datasets_with_dedup(task, rank, eval_split: str = "test"):
    """
    Load MTEB-style dataset with deduplication against evaluation split.
    Removes any training queries that appear in the evaluation set.
    """
    if rank == 0:
        print("Loading datasets with deduplication...")
        print(f"Loading eval queries from {task.hf_name} ({eval_split} split)...")
    
    # Load eval queries for deduplication
    eval_query_texts = get_mteb_eval_queries(task.hf_name, eval_split)
    
    if rank == 0:
        print(f"Found {len(eval_query_texts)} eval queries to exclude")
    
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
    excluded_count = 0

    for qrel in qrels:
        anchor_id = qrel[task.qrels_fields["anchor_id"]]
        positive_id = qrel[task.qrels_fields["positive_id"]]
        score = qrel[task.qrels_fields["score"]]

        # Filter invalid pairs
        if anchor_id not in queries_dict or positive_id not in corpus_dict or score < 1:
            continue

        query_text = queries_dict[anchor_id]
        
        # Skip if query appears in eval set
        if normalize_text(query_text) in eval_query_texts:
            excluded_count += 1
            continue

        # Extract query
        query_ids.append(anchor_id)
        query_texts.append(query_text)

        # Extract positive
        positive_entry = corpus_dict[positive_id]
        positive_ids.append(positive_id)

        if has_title:
            text, title = positive_entry
            positive_texts.append(text)
            positive_titles.append(title)
        else:
            positive_texts.append(positive_entry)

        # queries can be repeated many times in the search
        # for negatives we just want unique queries
        if anchor_id not in seen_queries:
            seen_queries.add(anchor_id)
            unique_query_ids.append(anchor_id)
            unique_query_texts.append(query_text)
            unique_positive_ids.append(positive_id)

            if has_title:
                text, title = positive_entry
                unique_positive_texts.append(text)
                unique_positive_titles.append(title)
            else:
                unique_positive_texts.append(positive_entry)

    if rank == 0:
        print(f"Excluded {excluded_count} query-doc pairs due to eval overlap")

    # Extract all documents from corpus
    document_ids = list(corpus_dict.keys())
    if has_title:
        document_texts = [text for text, title in corpus_dict.values()]
        document_titles = [title for text, title in corpus_dict.values()]
    else:
        document_texts = list(corpus_dict.values())
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


def from_multiple_hf_datasets(task, rank):

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

    for qrel in qrels:
        anchor_id = qrel[task.qrels_fields["anchor_id"]]
        positive_id = qrel[task.qrels_fields["positive_id"]]
        score = qrel[task.qrels_fields["score"]]

        # Filter invalid pairs
        if anchor_id not in queries_dict or positive_id not in corpus_dict or score < 1:
            continue

        # Extract query
        query_ids.append(anchor_id)
        query_texts.append(queries_dict[anchor_id])

        # Extract positive
        positive_entry = corpus_dict[positive_id]
        positive_ids.append(positive_id)

        if has_title:
            text, title = positive_entry
            positive_texts.append(text)
            positive_titles.append(title)
        else:
            positive_texts.append(positive_entry)

        # queries can be repeted many times in the search
        # for negatives we just want unique queries
        if anchor_id not in seen_queries:
            seen_queries.add(anchor_id)
            unique_query_ids.append(anchor_id)
            unique_query_texts.append(queries_dict[anchor_id])
            unique_positive_ids.append(positive_id)

            if has_title:
                text, title = positive_entry
                unique_positive_texts.append(text)
                unique_positive_titles.append(title)
            else:
                unique_positive_texts.append(positive_entry)

    # Extract all documents from corpus
    document_ids = list(corpus_dict.keys())
    if has_title:
        document_texts = [text for text, title in corpus_dict.values()]
        document_titles = [title for text, title in corpus_dict.values()]
    else:
        document_texts = list(corpus_dict.values())
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


def load_nli_retrieval(task) -> RetrievalRawData:
    """Load NLI datasets (SNLI, MNLI, ANLI) for retrieval.

    Filters for entailment pairs where premise is query and hypothesis is positive.
    """
    dataset = load_dataset(task.hf_name, split=task.split)

    # Filter for entailment pairs only (label == entailment_label)
    entailment_label = getattr(task, "entailment_label", 0)
    dataset = dataset.filter(lambda x: x[task.label_name] == entailment_label)

    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])
    document_texts = positive_texts.copy()

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_arguana_dedup_retrieval(task) -> RetrievalRawData:
    """Load BeIR/arguana-generated-queries with deduplication against mteb/arguana.

    This removes any query-positive pairs where either the query or positive text
    appears in the mteb/arguana evaluation set, preventing train-test contamination.
    """
    # Load the BeIR arguana dataset
    dataset = load_dataset(task.hf_name, split=task.split)

    # Get MTEB arguana texts for deduplication
    print("Loading mteb/arguana for deduplication...")
    mteb_query_texts, mteb_corpus_texts = get_mteb_arguana_texts()
    print(
        f"Found {len(mteb_query_texts)} MTEB queries and {len(mteb_corpus_texts)} MTEB corpus texts"
    )

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
    print(
        f"Removed {original_size - filtered_size} overlapping examples ({original_size} -> {filtered_size})"
    )

    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])
    document_texts = positive_texts.copy()

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_squad_retrieval(task) -> RetrievalRawData:
    """Load SQuAD dataset for retrieval.

    Question is used as query, context is used as positive document.
    """
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])

    # Remove duplicate contexts to create corpus
    seen_contexts = {}
    document_texts = []
    document_ids = []
    positive_ids = []

    for i, context in enumerate(positive_texts):
        if context not in seen_contexts:
            doc_id = f"doc_{len(seen_contexts)}"
            seen_contexts[context] = doc_id
            document_texts.append(context)
            document_ids.append(doc_id)
        positive_ids.append(seen_contexts[context])

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_stackexchange_retrieval(task) -> RetrievalRawData:
    """Load StackExchange dataset for retrieval.

    Title+body is used as query, upvoted_answer is used as positive document.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    # Combine title and body for query
    query_texts = [f"{row['title']} {row['body']}" for row in dataset]
    positive_texts = list(dataset[task.positive_name])
    document_texts = positive_texts.copy()

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


MIRACL_LANGUAGES = ["ar", "bn", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "yo", "de",]


def load_miracl_retrieval(task) -> RetrievalRawData:
    """Load MIRACL multilingual retrieval dataset.
    
    If hf_subset is None, loads all available languages.
    """
    query_texts = []
    positive_texts = []
    
    if task.hf_subset:
        # Load single language
        languages = [task.hf_subset]
    else:
        # Load all languages
        languages = MIRACL_LANGUAGES
        print(f"Loading MIRACL for all {len(languages)} languages...")
    
    for lang in languages:
        try:
            dataset = load_dataset(task.hf_name, name=lang, split=task.split)
            for row in dataset:
                query = row["query"]
                # positive_passages is a list of dicts with 'text' field
                positives = row.get("positive_passages", [])
                if positives and len(positives) > 0:
                    query_texts.append(query)
                    # Take the first positive passage
                    positive_texts.append(positives[0].get("text", ""))
            if not task.hf_subset:
                print(f"  Loaded {lang}: {len(dataset)} samples")
        except Exception as e:
            print(f"  Skipping {lang}: {e}")
            continue

    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_pubmedqa_retrieval(task) -> RetrievalRawData:
    """Load PubMedQA dataset for retrieval.

    Question is query, context (long_answer) is positive document.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = []
    positive_texts = []

    for row in dataset:
        question = row.get("question", "")
        # Context can be a list or dict with 'contexts' field
        context = row.get("context", {})
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if contexts:
                positive_text = " ".join(contexts)
            else:
                continue
        elif isinstance(context, list):
            positive_text = " ".join(context)
        elif isinstance(context, str):
            positive_text = context
        else:
            continue

        if question and positive_text:
            query_texts.append(question)
            positive_texts.append(positive_text)

    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_xsum_retrieval(task) -> RetrievalRawData:
    """Load XSum summarization dataset for retrieval.

    Summary is used as query, document is used as positive.
    """
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset["summary"])
    positive_texts = list(dataset["document"])
    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_cnndm_retrieval(task) -> RetrievalRawData:
    """Load CNN/DailyMail summarization dataset for retrieval.

    Highlights (summary) is used as query, article is used as positive.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset["highlights"])
    positive_texts = list(dataset["article"])
    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_stackoverflow_dup_retrieval(task) -> RetrievalRawData:
    """Load StackOverflow duplicate questions dataset for retrieval."""
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = []
    positive_texts = []

    for row in dataset:
        query = row.get("query", "")
        # positive can be a list of positive examples
        positives = row.get("positive", [])
        if isinstance(positives, list) and len(positives) > 0:
            query_texts.append(query)
            positive_texts.append(positives[0])
        elif isinstance(positives, str) and positives:
            query_texts.append(query)
            positive_texts.append(positives)

    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_sts_retrieval(task) -> RetrievalRawData:
    """Load STS datasets for retrieval.

    Treats sentence1 as query and sentence2 as positive document.
    Filters for high similarity scores (>= 4.0 on 0-5 scale).
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    score_name = getattr(task, "score_name", "score")
    score_threshold = getattr(task, "score_threshold", 4.0)

    query_texts = []
    positive_texts = []

    for row in dataset:
        score = row.get(score_name, 0)
        if score >= score_threshold:
            query_texts.append(row[task.anchor_name])
            positive_texts.append(row[task.positive_name])

    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
    )
