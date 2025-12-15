import torch
import torch.distributed as dist
from mteb import MTEB
from datasets import load_dataset
from torch.utils.data import Dataset

from mteb import get_task
import json


class MTEBDataset(Dataset):
    """Wrapper for MTEB datasets compatible with PyTorch DDP."""

    def __init__(self, data, task_type="retrieval"):
        self.data = data
        self.task_type = task_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Return format depends on task type
        if self.task_type == "retrieval":
            return {
                "query": item.get("query", ""),
                "positive": item.get("positive", ""),
                "negative": item.get("negative", ""),
            }
        elif self.task_type == "classification":
            return {"text": item.get("text", ""), "label": item.get("label", 0)}
        else:  # clustering, reranking, etc.
            return item


def load_mteb_task(task_name, split="test"):
    """Load a specific MTEB task dataset.

    Args:
        task_name: Name of the MTEB task (e.g., 'NFCorpus', 'SCIDOCS')
        split: Dataset split to load ('train', 'dev', 'test')

    Returns:
        dataset: Loaded dataset
        task_metadata: Task information
    """
    # Load MTEB task
    mteb = MTEB(tasks=[task_name])

    task = mteb.tasks[0]

    # Load the dataset
    task.load_data()

    # Extract the appropriate split
    if hasattr(task, "dataset") and task.dataset:
        if split in task.dataset:
            data = task.dataset[split]
        else:
            available = list(task.dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")
    else:
        raise ValueError(f"No dataset loaded for task {task_name}")

    task_metadata = {
        "name": task.metadata.name,
        "type": task.metadata.type,
        "description": task.description,
    }

    return data, task_metadata


def extract_mteb_data(task_name):
    """
    Extracts the dataset and necessary evaluation metadata from a given MTEB task.
    """

    # 1. Load the task object
    task = get_task(task_name)

    # 2. Extract metadata
    metadata = {
        k: v
        for k, v in task.metadata.__dict__.items()
        if not k.startswith("_")  # Clean up private attributes
    }

    # 3. Load the actual data
    # Only load the splits defined for evaluation (e.g., ['test'])
    task.load_data(eval_splits=task.metadata.eval_splits)

    extracted_datasets = {}
    for split_name, dataset in task.dataset.items():
        # Convert Hugging Face Dataset to a standard list of dicts for export
        # This makes it easy to integrate into any external pipeline
        extracted_datasets[split_name] = dataset.to_list()
        print(f"Extracted split '{split_name}' with {len(dataset)} samples.")

    return {"metadata": metadata, "data": extracted_datasets}


# Example Usage:


task_name = "Banking77Classification.v2"
task = get_task(task_name)

# 2. Extract metadata
metadata = {
    k: v
    for k, v in task.metadata.__dict__.items()
    if not k.startswith("_")  # Clean up private attributes
}


task.load_data()

task.dataset["test"]

import inspect

sig = inspect.signature(task.load_data)

print(sig)


extracted_info = extract_mteb_data("Banking77Classification.v2")


task_name = "NFCorpus"  # Example retrieval task


mteb = MTEB(tasks=[task_name])

task = mteb.tasks[0]

# Load the dataset
task.load_data()
data, metadata = load_mteb_task(task_name, split="test")
