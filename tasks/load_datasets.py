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

    # --- Old implementation (single call, higher peak memory) ---
    # corpus_ds = dict_to_dataset(
    #     texts=raw_data.document_texts,
    #     ids=raw_data.document_ids,
    #     titles=raw_data.document_titles,
    # )
    # del raw_data.document_texts, raw_data.document_ids, raw_data.document_titles
    # _print_ram("after dict_to_dataset (corpus)", rank)

    # Build each Arrow column one at a time, freeing the Python list from
    # raw_data before allocating the next one.  This avoids the peak where
    # all three Python lists AND all three Arrow arrays coexist in memory.
    arr_doc_text = pa.array(raw_data.document_texts, type=pa.string())
    del raw_data.document_texts
    gc.collect()
    _print_ram("after arr_doc_text", rank)

    arr_doc_id = pa.array(raw_data.document_ids, type=pa.string())
    del raw_data.document_ids
    gc.collect()
    _print_ram("after arr_doc_id", rank)

    has_titles = raw_data.document_titles is not None
    if has_titles:
        arr_doc_title = pa.array(raw_data.document_titles, type=pa.string())
        del raw_data.document_titles
        gc.collect()
        _print_ram("after arr_doc_title", rank)
        names = ["text", "id", "title"]
        arrays = [arr_doc_text, arr_doc_id, arr_doc_title]
    else:
        del raw_data.document_titles
        names = ["text", "id"]
        arrays = [arr_doc_text, arr_doc_id]

    # Write to a temp Arrow IPC file (zero-copy streaming from existing buffers),
    # then free the source arrays and memory-map the file back.
    # This avoids the in-process copy that Dataset(pa.table(...)) may trigger
    # when combining large chunked string columns (which caused the OOM/hang).
    schema = pa.schema([(n, pa.string()) for n in names])
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".arrow")
    os.close(tmp_fd)
    if rank == 0 and verbose:
        print(f"Writing corpus Arrow file to {tmp_path}")
    _print_ram("before ipc write", rank)
    with pa_ipc.new_file(tmp_path, schema) as writer:
        writer.write_table(pa.table(dict(zip(names, arrays))))
    _print_ram("after ipc write", rank)

    # Free the in-memory arrays before opening the mmap'd dataset
    del arrays, arr_doc_text, arr_doc_id
    if has_titles:
        del arr_doc_title
    gc.collect()
    _print_ram("after del arrays (before mmap)", rank)

    # Memory-map the IPC file: buffers are backed by disk pages, not RAM.
    # On Linux, unlinking after mmap is safe — the data remains accessible.
    mm = pa.memory_map(tmp_path, "r")
    corpus_table = pa.ipc.open_file(mm).read_all()
    os.unlink(tmp_path)
    corpus_ds = Dataset(corpus_table)
    del corpus_table
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
