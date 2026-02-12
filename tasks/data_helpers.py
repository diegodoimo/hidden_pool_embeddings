from datasets import load_dataset, Dataset, Features, Value
import time
import os
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Sequence
import torch.distributed as dist


def process_chunk(args):
    chunk, id_field, text_field, title_field = args
    if title_field:
        return {
            row[id_field]: {"text": row[text_field], "title": row[title_field]}
            for row in chunk
        }
    else:
        return {row[id_field]: {"text": row[text_field]} for row in chunk}


def get_dict(dataset, id_field, text_field, title_field=None):
    cpu_count = os.cpu_count() or 8
    if dist.is_initialized():
        n_workers = max(1, cpu_count // dist.get_world_size())
    else:
        n_workers = min(16, cpu_count)
    chunk_size = max(1, len(dataset) // n_workers)

    chunks = [
        dataset.select(range(i, min(i + chunk_size, len(dataset))))
        for i in range(0, len(dataset), chunk_size)
    ]

    with Pool(n_workers) as pool:
        results = pool.map(
            process_chunk,
            [(chunk, id_field, text_field, title_field) for chunk in chunks],
        )

    # Merge dictionaries
    return {k: v for d in results for k, v in d.items()}


def dict_to_dataset(texts, ids, titles=None, ids_only=False):
    """Create a HuggingFace dataset from texts and IDs.

    Args:
        texts: List of text strings (ignored if ids_only=True)
        ids: List of ID strings
        titles: Optional list of title strings (ignored if ids_only=True)
        ids_only: If True, create dataset with only IDs (no texts or titles)
    """
    if ids_only:
        # Create dataset with only IDs (for queries/positives with repetitions)
        dataset = Dataset.from_dict(
            {"id": ids},
            features=Features({"id": Value("string")}),
        )
    elif titles is not None:
        dataset = Dataset.from_dict(
            {
                "text": texts,
                "id": ids,
                "title": titles,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                    "title": Value("string"),
                }
            ),
        )
    else:
        dataset = Dataset.from_dict(
            {
                "text": texts,
                "id": ids,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                }
            ),
        )

    return dataset


def create_qrels_dataset(query_ids, positive_ids):
    """Create a qrels dataset with query_id and positive_id columns.

    Args:
        query_ids: List of query ID strings
        positive_ids: List of positive ID strings (must be same length as query_ids)

    Returns:
        Dataset with query_id and positive_id columns
    """
    assert len(query_ids) == len(
        positive_ids
    ), "query_ids and positive_ids must have the same length"

    dataset = Dataset.from_dict(
        {
            "query_id": query_ids,
            "positive_id": positive_ids,
        },
        features=Features(
            {
                "query_id": Value("string"),
                "positive_id": Value("string"),
            }
        ),
    )
    return dataset


@dataclass
class RetrievalRawData:
    """Raw data structure for retrieval tasks (includes STS tasks treated as retrieval)."""

    query_ids: List[str]
    positive_ids: List[str]
    positive_titles: Optional[Sequence[str]]

    document_texts: Sequence[str]  # List[str] or pd.Series
    document_ids: List[str]
    document_titles: Optional[Sequence[str]]

    unique_query_texts: Sequence[str]  # List[str] or pd.Series
    unique_query_ids: List[str]

    unique_positive_texts: Sequence[str]  # List[str] or pd.Series
    unique_positive_ids: List[str]
    unique_positive_titles: Optional[Sequence[str]]

    corpus_dict: Dict[str, Dict[str, str]]
    has_title: bool
    documents_are_positives: bool


@dataclass
class ClassificationRawData:
    """Raw data structure for classification and clustering tasks."""

    texts: List[str]
    labels: List[int]
    ids: Optional[List[str]] = None
