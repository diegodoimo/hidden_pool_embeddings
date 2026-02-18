from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


MIRACL_LANGUAGES = [
    "ar",
    "bn",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "sw",
    "te",
    "th",
    "zh",
    "yo",
    "de",
]


def extract_unique_queries(
    query_texts: list[str],
    query_ids: list[str],
    positive_texts: list[str],
    positive_ids: list[str],
    positive_titles: list[str] | None = None,
) -> tuple:
    """Extract unique queries from lists that may contain repeated queries."""
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


def load_miracl_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """Load MIRACL multilingual retrieval dataset using vectorized operations.

    If hf_subset is None, loads all available languages.
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset(s)...")

    query_texts = []
    positive_texts = []

    if task.hf_subset:
        # Load single language
        languages = [task.hf_subset]
    else:
        # Load all languages
        languages = MIRACL_LANGUAGES
        if rank == 0:
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
            if not task.hf_subset and rank == 0:
                print(f"  Loaded {lang}: {return_formatted(len(dataset))} samples")
        except Exception as e:
            if rank == 0:
                print(f"  Skipping {lang}: {e}")
            continue

    dist.barrier()
    if rank == 0:
        print(f"Dataset(s) loaded in {(time.time()-start)/60:.2f} min")
        print(f"Total pairs extracted: {return_formatted(len(query_texts))}")
        start = time.time()
        print("Converting to pandas...")

    # Convert to pandas DataFrame
    df = pd.DataFrame({"query": query_texts, "positive": positive_texts})
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


class MIRACL(AbsTask):
    """MIRACL multilingual retrieval dataset.

    Note: Each language is a separate config. Set hf_subset to specific language
    (ar, bn, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, zh, yo, de)
    or None to load all languages (requires custom loader modification).
    """

    language = "multilingual"

    hf_name = "miracl/miracl"
    hf_subset = None  # Load all available languages
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "positive_passages"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    loader = load_miracl_retrieval
