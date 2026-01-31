from datasets import load_dataset

from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value
import time
import os
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Optional, Dict, Set


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
    n_workers = 16
    chunk_size = len(dataset) // n_workers

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


def dict_to_dataset(texts, ids, titles=None):

    if titles is not None:

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


@dataclass
class RetrievalRawData:
    query_texts: List[str]
    query_ids: List[str]

    positive_texts: List[str]
    positive_ids: List[str]
    positive_titles: Optional[List[str]]

    document_texts: List[str]
    document_ids: List[str]
    document_titles: Optional[List[str]]

    unique_query_texts: List[str]
    unique_query_ids: List[str]

    unique_positive_texts: List[str]
    unique_positive_ids: List[str]
    unique_positive_titles: Optional[List[str]]

    corpus_dict: Dict[str, Dict[str, str]]
    has_title: bool
