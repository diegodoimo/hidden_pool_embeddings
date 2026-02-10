"""
Multi-way classification-specific loader functions.
These loaders handle multi-way text classification tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import random
from collections import defaultdict

from tasks.data_helpers import RetrievalRawData, ClassificationRawData


def load_classification_standard(task, rank=0):
    """
    Standard loader for classification tasks (legacy - simple format).
    Loads data from a single HuggingFace dataset with texts and labels.
    
    Returns ClassificationRawData format.
    Used by tasks that still need simple classification format.
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


def load_multiway_classification_sampling(task, rank=0, num_hard_negatives=24):
    """
    Load multi-way classification datasets with in-batch sampling strategy.
    
    For each query:
    - A random sample from the same class is used as its positive passage
    - num_hard_negatives samples from other classes are selected as hard negatives
    
    Args:
        task: Task object with standard attributes
        rank: Process rank for logging
        num_hard_negatives: Number of hard negatives from other classes (default: 24)
    
    Returns:
        RetrievalRawData with sampled positives and negatives in corpus
    
    Used by: Multi-way classification tasks
    """
    # Load dataset
    if hasattr(task, 'hf_subset') and task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)
    
    # Get label field name
    label_field = getattr(task, "label_name", None) or getattr(task, "label", "label")
    
    # Group texts by label
    label_to_texts = defaultdict(list)
    all_texts = []
    all_labels = []
    
    for row in dataset:
        text = row[task.anchor_name]
        label = row[label_field]
        all_texts.append(text)
        all_labels.append(label)
        label_to_texts[label].append(text)
    
    # Get all unique labels
    unique_labels = sorted(label_to_texts.keys())
    
    if rank == 0:
        print(f"Found {len(unique_labels)} classes in multi-way classification")
        for label in unique_labels:
            print(f"  Class {label}: {len(label_to_texts[label])} samples")
    
    # Build query-positive pairs
    query_texts = []
    query_ids = []
    positive_texts = []
    positive_ids = []
    
    # Create text-to-id mapping for corpus
    text_to_id = {}
    for idx, text in enumerate(all_texts):
        if text not in text_to_id:
            text_to_id[text] = f"doc_{len(text_to_id)}"
    
    # Create unique query mapping
    unique_query_texts = []
    unique_query_ids = []
    text_to_query_id = {}
    
    # For each text, create a query-positive pair
    for text, label in zip(all_texts, all_labels):
        # Get texts with same label
        same_label_texts = [t for t in label_to_texts[label] if t != text]
        
        # If there are other texts with same label, pick one as positive
        # Otherwise, use the text itself as positive
        if same_label_texts:
            positive_text = random.choice(same_label_texts)
        else:
            positive_text = text
        
        # Create unique query ID
        if text not in text_to_query_id:
            query_id = f"query_{len(unique_query_ids)}"
            text_to_query_id[text] = query_id
            unique_query_ids.append(query_id)
            unique_query_texts.append(text)
        else:
            query_id = text_to_query_id[text]
        
        query_texts.append(text)
        query_ids.append(query_id)
        positive_texts.append(positive_text)
        positive_ids.append(text_to_id[positive_text])
    
    # Build corpus: includes all texts
    # During hard negative mining, the system will select hard negatives from other classes
    document_texts = list(text_to_id.keys())
    document_ids = list(text_to_id.values())
    
    corpus_dict = {
        doc_id: {"text": doc_text} 
        for doc_id, doc_text in zip(document_ids, document_texts)
    }
    
    # Build unique positives
    unique_positive_texts = []
    unique_positive_ids = []
    seen_positives = set()
    
    for pos_text in positive_texts:
        if pos_text not in seen_positives:
            seen_positives.add(pos_text)
            unique_positive_texts.append(pos_text)
            unique_positive_ids.append(text_to_id[pos_text])
    
    if rank == 0:
        print(f"Loaded {len(query_texts)} query-positive pairs for multi-way classification")
        print(f"Corpus size: {len(document_texts)} unique texts")
        print(f"Hard negatives will be mined from corpus during training")
    
    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=None,
        corpus_dict=corpus_dict,
        has_title=False,
    )


def load_multiway_classification_hard_negatives(task, rank=0):
    """
    Load multi-way classification datasets for hard negative mining.
    
    Similar to retrieval tasks, this loader creates a corpus of all texts,
    allowing for hard negative mining. Each text is a query, and texts with
    the same label are treated as positives for each other.
    
    This is an alias for the sampling version since they both support hard negative mining.
    
    Args:
        task: Task object with standard attributes
        rank: Process rank for logging
    
    Returns:
        RetrievalRawData with all texts as corpus for mining
    
    Used by: Multi-way classification tasks when use_hard_negative_mining=True
    """
    return load_multiway_classification_sampling(task, rank=rank)
