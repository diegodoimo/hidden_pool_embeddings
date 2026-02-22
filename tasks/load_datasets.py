from datasets import load_dataset

from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value
import time
import os
import gc
import tempfile
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Union, Tuple
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from tasks.data_helpers import (
    dict_to_dataset,
    create_qrels_dataset,
    RetrievalRawData,
    ClassificationRawData,
    get_dict,
)
from tasks.retrieval_loaders import _print_ram
import torch.distributed as dist


def normalize_text(
    text: str,
) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def load_task_data(
    task, subtask=None, max_num_queries=None
) -> Union[Tuple[Dataset, Dict, bool, int], ClassificationRawData]:
    """
    Unified data loading function for all task types (retrieval, STS, classification, clustering).

    Args:
        task: Task object with metadata and configuration
        subtask: Optional subtask name for datasets with multiple subtasks
        max_num_queries: Maximum number of queries to load (default: 1 million)

    Returns:
        For retrieval/STS tasks: tuple of (hf_dataset, corpus_dict, has_title, n_positives)
            where hf_dataset contains: unique_queries, qrels, corpus
        For classification/clustering tasks: ClassificationRawData with texts, labels, and ids
    """
    task_type = task.metadata.type

    if task_type == "Retrieval":
        # Handle all retrieval and STS tasks (STS is treated as retrieval)
        return _load_retrieval_data(
            task=task, max_num_queries=max_num_queries, subtask=subtask
        )
    elif task_type in ["Classification", "BinaryClassification", "Clustering"]:
        # Handle classification, binary classification and clustering tasks.
        # When the task uses hard-negative mining its loader returns
        # RetrievalRawData, so we route it through the retrieval pipeline.
        use_hn = getattr(task, "use_hard_negative_mining", False)
        if use_hn:
            return _load_retrieval_data(
                task=task, max_num_queries=max_num_queries, subtask=subtask
            )
        return _load_classification_data(task)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _load_retrieval_data(
    task, subtask=None, max_num_queries=10**6
) -> Tuple[Dataset, Dict, bool, int]:
    """
    Load data for retrieval tasks (including STS tasks).

    Args:
        task: Task object with metadata and configuration
        subtask: Optional subtask name for datasets with multiple subtasks
        max_num_queries: Maximum number of queries to load (default: 1 million)

    Returns:
        tuple of (hf_dataset, corpus_dict, has_title, n_positives)
    """
    # Dispatch to appropriate loader based on task configuration
    rank = dist.get_rank()

    loader_func = getattr(task, "loader", None)

    if loader_func is None:
        raise ValueError(
            f"Task {task.__class__.__name__} does not have a 'loader' attribute defined. "
            "Please define a loader for this task."
        )

    raw_data = loader_func(task=task, subtask=subtask, max_num_queries=max_num_queries)

    # Convert raw data to HuggingFace datasets
    verbose = False
    if len(raw_data.query_ids) > 5 * 10**5:
        verbose = True

    if rank == 0 and verbose:
        print(f"Building qrels dataset")
    # Create qrels dataset with query_id and positive_id pairs
    qrels_ds = create_qrels_dataset(
        query_ids=raw_data.query_ids,
        positive_ids=raw_data.positive_ids,
    )
    # Free source lists right after Arrow conversion to reduce peak memory
    del raw_data.query_ids, raw_data.positive_ids
    _print_ram("after create_qrels_dataset", rank)

    if rank == 0 and verbose:
        print(f"Building queries dataset")
    unique_queries_ds = dict_to_dataset(
        texts=raw_data.unique_query_texts, ids=raw_data.unique_query_ids
    )
    del raw_data.unique_query_texts, raw_data.unique_query_ids
    _print_ram("after dict_to_dataset (queries)", rank)

    if rank == 0 and verbose:
        print(f"Building document dataset")

    corpus_ds = _build_corpus_dataset(
        texts=raw_data.document_texts,
        ids=raw_data.document_ids,
        titles=raw_data.document_titles,
        rank=rank,
        verbose=verbose,
    )
    del raw_data.document_texts, raw_data.document_ids, raw_data.document_titles
    _print_ram("after dict_to_dataset (corpus)", rank)

    hf_dataset = {
        "unique_queries": unique_queries_ds,
        "qrels": qrels_ds,
        "corpus": corpus_ds,
    }

    return (
        hf_dataset,
        raw_data.corpus_dict,
        raw_data.query_dict,
        raw_data.has_title,
        raw_data.n_positives,
    )


def _load_classification_data(task) -> ClassificationRawData:
    """
    Load data for classification and clustering tasks (sampling mode).

    Returns:
        ClassificationRawData with texts, labels, and ids
    """
    # Every classification/clustering task should have a 'loader' attribute
    loader_func = getattr(task, "loader", None)

    if loader_func is None:
        raise ValueError(
            f"Task {task.__class__.__name__} does not have a 'loader' attribute defined. "
            "Please define a loader for this task."
        )

    # Call the loader function (sampling loaders accept task + rank)
    return loader_func(task)


# Helper function for building corpus dictionaries (used by loaders)
def _build_corpus_dict(dataset, id_field, text_field, title_field=None):
    """Helper to build corpus dictionary from dataset."""
    return get_dict(dataset, id_field, text_field, title_field)


# Threshold above which the mmap IPC strategy is used instead of the
# standard in-memory pa.table() path.  The large datasets that require
# it (s2orc_*, paq, bioasq) all have >9M documents; normal datasets are
# well below 5M.
_LARGE_CORPUS_THRESHOLD = 5_000_000


def _build_corpus_dataset(texts, ids, titles, rank=0, verbose=False) -> Dataset:
    """Build a HuggingFace Dataset from corpus lists.

    For small corpora (< _LARGE_CORPUS_THRESHOLD documents) uses the fast
    in-memory pa.table() path via dict_to_dataset.

    For large corpora (>= _LARGE_CORPUS_THRESHOLD documents) uses a
    write-then-mmap strategy to avoid the combine_chunks() copy that
    Dataset(pa.table(...)) triggers on ChunkedArrays produced by pa.array()
    over lists exceeding Arrow's 2 GB buffer limit.  That copy can spike
    memory by 30-60 GB per process and cause OOM on datasets like
    s2orc_title_abstract (~41M docs), paq (~9M docs), and bioasq (~14M docs).

    The mmap strategy:
      1. Convert each Python list to an Arrow array one at a time, freeing
         the source list before converting the next (avoids Python+Arrow
         double-presence during conversion).
      2. Write all columns to a temp Arrow IPC file — the IPC writer
         handles ChunkedArrays natively with zero in-memory copy.
      3. Delete all in-memory Arrow arrays and gc.collect().
      4. Memory-map the file back; the resulting table's buffers are backed
         by OS file pages, not heap RAM.
      5. Wrap in Dataset() — no combine_chunks() is triggered because the
         table is already in contiguous on-disk layout.
      6. Unlink the temp file (safe on Linux: mmap keeps the inode alive).
    """
    n_docs = len(ids)

    if n_docs < _LARGE_CORPUS_THRESHOLD:
        # Fast in-memory path for normal-sized corpora.
        return dict_to_dataset(texts=texts, ids=ids, titles=titles)

    # --- Large-corpus mmap path ---
    if rank == 0 and verbose:
        print(
            f"Large corpus detected ({n_docs:,} docs >= {_LARGE_CORPUS_THRESHOLD:,}): "
            "using write-then-mmap strategy"
        )

    arr_text = pa.array(texts, type=pa.string())
    del texts
    gc.collect()
    _print_ram("after arr_doc_text", rank)

    arr_id = pa.array(ids, type=pa.string())
    del ids
    gc.collect()
    _print_ram("after arr_doc_id", rank)

    has_titles = titles is not None
    if has_titles:
        arr_title = pa.array(titles, type=pa.string())
        del titles
        gc.collect()
        _print_ram("after arr_doc_title", rank)
        names = ["text", "id", "title"]
        arrays = [arr_text, arr_id, arr_title]
    else:
        names = ["text", "id"]
        arrays = [arr_text, arr_id]

    schema = pa.schema([(n, pa.string()) for n in names])
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".arrow")
    os.close(tmp_fd)
    if rank == 0 and verbose:
        print(f"Writing corpus Arrow file to {tmp_path}")
    _print_ram("before ipc write", rank)
    with pa_ipc.new_file(tmp_path, schema) as writer:
        writer.write_table(pa.table(dict(zip(names, arrays))))
    _print_ram("after ipc write", rank)

    del arrays, arr_text, arr_id
    if has_titles:
        del arr_title
    gc.collect()
    _print_ram("after del arrays (before mmap)", rank)

    mm = pa.memory_map(tmp_path, "r")
    corpus_table = pa_ipc.open_file(mm).read_all()
    os.unlink(tmp_path)
    corpus_ds = Dataset(corpus_table)
    del corpus_table
    return corpus_ds
