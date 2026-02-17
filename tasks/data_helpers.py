from datasets import load_dataset, Dataset, Features, Value
import pyarrow as pa
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

    Uses PyArrow arrays directly to avoid the double-copy overhead of
    Dataset.from_dict (Python list -> dict -> Arrow).  This is 3-5x faster
    and uses ~50% less peak memory for large datasets (e.g. 50M rows).

    Args:
        texts: List/Series of text strings (ignored if ids_only=True)
        ids: List/Series of ID strings
        titles: Optional list/Series of title strings (ignored if ids_only=True)
        ids_only: If True, create dataset with only IDs (no texts or titles)
    """
    if ids_only:
        table = pa.table({"id": pa.array(ids, type=pa.string())})
    elif titles is not None:
        table = pa.table(
            {
                "text": pa.array(texts, type=pa.string()),
                "id": pa.array(ids, type=pa.string()),
                "title": pa.array(titles, type=pa.string()),
            }
        )
    else:
        table = pa.table(
            {
                "text": pa.array(texts, type=pa.string()),
                "id": pa.array(ids, type=pa.string()),
            }
        )

    return Dataset(table)


# --- Previous (slower) implementation kept for reference ---
# def dict_to_dataset(texts, ids, titles=None, ids_only=False):
#     if ids_only:
#         dataset = Dataset.from_dict(
#             {"id": ids},
#             features=Features({"id": Value("string")}),
#         )
#     elif titles is not None:
#         dataset = Dataset.from_dict(
#             {"text": texts, "id": ids, "title": titles},
#             features=Features(
#                 {"text": Value("string"), "id": Value("string"), "title": Value("string")}
#             ),
#         )
#     else:
#         dataset = Dataset.from_dict(
#             {"text": texts, "id": ids},
#             features=Features({"text": Value("string"), "id": Value("string")}),
#         )
#     return dataset


# --- Alternative 1: from_generator (streaming construction) ---
# Advantage: Streams rows into Arrow in small batches (writer_batch_size),
# so the full Python list and the full Arrow table never coexist in memory.
# Best when the input data is VERY large and you are memory-constrained
# (e.g. close to OOM). Slightly slower than the PyArrow approach above
# because it iterates row-by-row in Python, but peak memory is minimal.
#
# def dict_to_dataset(texts, ids, titles=None, ids_only=False):
#     if ids_only:
#         features = Features({"id": Value("string")})
#         def gen():
#             for i in range(len(ids)):
#                 yield {"id": ids[i]}
#     elif titles is not None:
#         features = Features(
#             {"text": Value("string"), "id": Value("string"), "title": Value("string")}
#         )
#         def gen():
#             for i in range(len(ids)):
#                 yield {"text": texts[i], "id": ids[i], "title": titles[i]}
#     else:
#         features = Features({"text": Value("string"), "id": Value("string")})
#         def gen():
#             for i in range(len(ids)):
#                 yield {"text": texts[i], "id": ids[i]}
#     return Dataset.from_generator(
#         gen, features=features, writer_batch_size=10_000
#     )


# --- Alternative 2: Memory-mapped cache (disk-backed Arrow) ---
# Advantage: The Arrow table is written to a cache file on disk and
# memory-mapped, so it does NOT consume RAM. Ideal when you have plenty
# of disk but limited RAM, or when the same dataset is loaded repeatedly
# across runs (the cache is reused). Combines with from_generator for
# minimal peak memory. Slower on first load (disk I/O) but instant on
# subsequent loads if the cache exists.
#
# def dict_to_dataset(texts, ids, titles=None, ids_only=False,
#                     cache_dir="/tmp/hpe_cache"):
#     if ids_only:
#         features = Features({"id": Value("string")})
#         def gen():
#             for i in range(len(ids)):
#                 yield {"id": ids[i]}
#     elif titles is not None:
#         features = Features(
#             {"text": Value("string"), "id": Value("string"), "title": Value("string")}
#         )
#         def gen():
#             for i in range(len(ids)):
#                 yield {"text": texts[i], "id": ids[i], "title": titles[i]}
#     else:
#         features = Features({"text": Value("string"), "id": Value("string")})
#         def gen():
#             for i in range(len(ids)):
#                 yield {"text": texts[i], "id": ids[i]}
#     return Dataset.from_generator(
#         gen, features=features, writer_batch_size=10_000,
#         cache_dir=cache_dir, keep_in_memory=False,
#     )


def create_qrels_dataset(query_ids, positive_ids):
    """Create a qrels dataset with query_id and positive_id columns.

    Uses PyArrow arrays directly for efficient construction at scale.

    Args:
        query_ids: List of query ID strings
        positive_ids: List of positive ID strings (must be same length as query_ids)

    Returns:
        Dataset with query_id and positive_id columns
    """
    assert len(query_ids) == len(
        positive_ids
    ), "query_ids and positive_ids must have the same length"

    table = pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.string()),
            "positive_id": pa.array(positive_ids, type=pa.string()),
        }
    )
    return Dataset(table)


# --- Previous (slower) implementation kept for reference ---
# def create_qrels_dataset(query_ids, positive_ids):
#     assert len(query_ids) == len(positive_ids)
#     dataset = Dataset.from_dict(
#         {"query_id": query_ids, "positive_id": positive_ids},
#         features=Features(
#             {"query_id": Value("string"), "positive_id": Value("string")}
#         ),
#     )
#     return dataset


@dataclass
class RetrievalRawData:
    """Raw data structure for retrieval tasks (includes STS tasks treated as retrieval).

    Documents are organized with unique positives first (at indices 0 to n_positives-1),
    followed by other unique documents. This unified format allows efficient corpus
    construction regardless of whether documents come from one or multiple datasets.
    """

    query_ids: List[str]
    positive_ids: List[str]

    document_texts: Sequence[
        str
    ]  # List[str] or pd.Series - unique docs with positives first
    document_ids: List[str]  # IDs for unique documents with positives first
    document_titles: Optional[
        Sequence[str]
    ]  # Titles for unique documents (if available)

    unique_query_texts: Sequence[str]  # List[str] or pd.Series
    unique_query_ids: List[str]

    corpus_dict: Dict[str, Dict[str, str]]
    query_dict: Dict[str, Dict[str, str]]
    has_title: bool
    n_positives: int  # Number of unique positives at the beginning of documents


@dataclass
class ClassificationRawData:
    """Raw data structure for classification and clustering tasks."""

    texts: List[str]
    labels: List[int]
    ids: Optional[List[str]] = None
