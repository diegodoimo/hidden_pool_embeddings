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
    create_qrels_dataset,
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
    task, subtask=None, max_num_queries=10**6
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

    loader_func = getattr(task, "loader", None)

    if loader_func is None:
        raise ValueError(
            f"Task {task.__class__.__name__} does not have a 'loader' attribute defined. "
            "Please define a loader for this task."
        )

    raw_data = loader_func(task=task, subtask=subtask, max_num_queries=max_num_queries)

    # Convert raw data to HuggingFace datasets
    # Create qrels dataset with query_id and positive_id pairs

    qrels_ds = create_qrels_dataset(
        query_ids=raw_data.query_ids,
        positive_ids=raw_data.positive_ids,
    )
    # Free source lists right after Arrow conversion to reduce peak memory
    del raw_data.query_ids, raw_data.positive_ids

    unique_queries_ds = dict_to_dataset(
        texts=raw_data.unique_query_texts, ids=raw_data.unique_query_ids
    )
    del raw_data.unique_query_texts, raw_data.unique_query_ids

    corpus_ds = dict_to_dataset(
        texts=raw_data.document_texts,
        ids=raw_data.document_ids,
        titles=raw_data.document_titles,
    )
    del raw_data.document_texts, raw_data.document_ids, raw_data.document_titles

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
