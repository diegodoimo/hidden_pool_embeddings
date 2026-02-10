from datasets import load_dataset

from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value
import time
import os
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Union, Tuple
from tasks.data_helpers import (
    dict_to_dataset,
    RetrievalRawData,
    ClassificationRawData,
    get_dict,
)
import torch.distributed as dist


def normalize_text(
    text: str,
) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def load_task_data(
    task) -> Union[Tuple[Dataset, Dict, bool], ClassificationRawData]:
    """
    Unified data loading function for all task types (retrieval, STS, classification, clustering).

    Args:
        task: Task object with metadata and configuration
        rank: Process rank for distributed training (default: 0)

    Returns:
        For retrieval/STS tasks: tuple of (hf_dataset, corpus_dict, has_title)
            where hf_dataset contains: unique_queries, unique_positives, queries, positives, corpus
        For classification/clustering tasks: ClassificationRawData with texts, labels, and ids
    """
    task_type = task.metadata.type
    rank = dist.get_rank()

    if task_type == "Retrieval":
        # Handle all retrieval and STS tasks (STS is treated as retrieval)
        return _load_retrieval_data(task, rank)
    elif task_type in ["Classification", "Clustering"]:
        # Handle classification and clustering tasks
        return _load_classification_data(task, rank)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _load_retrieval_data(task, rank=0) -> Tuple[Dataset, Dict, bool]:
    """
    Load data for retrieval tasks (including STS tasks).

    Returns:
        tuple of (hf_dataset, corpus_dict, has_title)
    """
    # Dispatch to appropriate loader based on task configuration
    raw_data = _get_retrieval_raw_data(task, rank)

    # Convert raw data to HuggingFace datasets
    queries_ds = dict_to_dataset(texts=raw_data.query_texts, ids=raw_data.query_ids)
    unique_queries_ds = dict_to_dataset(
        texts=raw_data.unique_query_texts, ids=raw_data.unique_query_ids
    )

    positives_ds = dict_to_dataset(
        texts=raw_data.positive_texts,
        ids=raw_data.positive_ids,
        titles=raw_data.positive_titles,
    )
    unique_positive_ds = dict_to_dataset(
        texts=raw_data.unique_positive_texts,
        ids=raw_data.unique_positive_ids,
        titles=raw_data.unique_positive_titles,
    )

    corpus_ds = dict_to_dataset(
        texts=raw_data.document_texts,
        ids=raw_data.document_ids,
        titles=raw_data.document_titles,
    )

    hf_dataset = {
        "unique_queries": unique_queries_ds,
        "unique_positives": unique_positive_ds,
        "queries": queries_ds,
        "positives": positives_ds,
        "corpus": corpus_ds,
    }

    return hf_dataset, raw_data.corpus_dict, raw_data.has_title


def _get_retrieval_raw_data(task, rank) -> RetrievalRawData:
    """
    Get raw retrieval data using the loader function defined in the task.
    Each task now has a 'loader' attribute that is the function to call.
    """
    # Every task should now have a 'loader' attribute
    loader_func = getattr(task, "loader", None)

    if loader_func is None:
        raise ValueError(
            f"Task {task.__class__.__name__} does not have a 'loader' attribute defined. "
            "Please define a loader for this task."
        )

    # Check if loader needs eval_split parameter (for dedup loaders)
    loader_name = loader_func.__name__
    if "with_dedup" in loader_name and hasattr(task, "eval_split"):
        # Pass rank and eval_split
        return loader_func(task, rank, task.eval_split)
    elif loader_name in [
        "from_multiple_hf_datasets",
        "from_multiple_hf_datasets_with_dedup",
    ]:
        # Pass rank parameter
        return loader_func(task, rank)
    else:
        # Call with just task
        return loader_func(task)


def _load_classification_data(task, rank) -> ClassificationRawData:
    """
    Load data for classification and clustering tasks.

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

    # Call the loader function
    return loader_func(task, rank)


# Helper function for building corpus dictionaries (used by loaders)
def _build_corpus_dict(dataset, id_field, text_field, title_field=None):
    """Helper to build corpus dictionary from dataset."""
    return get_dict(dataset, id_field, text_field, title_field)
