"""
Classification-specific loader functions.
These loaders handle text classification tasks.
"""

from datasets import load_dataset
from tasks.data_helpers import ClassificationRawData


def load_classification_standard(task, rank=0):
    """
    Standard loader for classification tasks.
    Loads data from a single HuggingFace dataset with texts and labels.
    
    Used by most classification tasks.
    """
    # Get label field name - could be "label", "label_name", or other custom field
    label_field = getattr(task, "label_name", None) or getattr(task, "label", "label")
    
    dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    
    texts = list(dataset[task.anchor_name])
    labels = list(dataset[label_field])
    ids = [f"doc_{i}" for i in range(len(texts))]
    
    if rank == 0:
        print(f"Loaded {len(texts)} samples for {task.metadata.type} task")
    
    return ClassificationRawData(texts=texts, labels=labels, ids=ids)
