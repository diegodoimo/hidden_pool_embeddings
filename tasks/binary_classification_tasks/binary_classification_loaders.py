"""
Binary classification-specific loader functions.
These loaders handle binary text classification tasks.
"""

from datasets import load_dataset
from typing import List, Optional, Dict
import random
from collections import defaultdict

from tasks.data_helpers import RetrievalRawData, ClassificationRawData


def load_binary_classification_label_based(task, rank=0):
    """
    Load binary classification datasets using label text as positives/negatives.

    For binary classification, each input is treated as a query, its label text
    (e.g., "toxic") as the positive passage, and the other class's label text
    (e.g., "not toxic") as one hard negative.

    Args:
        task: Task object with attributes:
            - hf_name: HuggingFace dataset name
            - hf_subset: Optional subset name
            - split: Dataset split
            - query_name: Text field name
            - label: Label field name
            - label_texts: Dict mapping label values to text (e.g., {0: "negative", 1: "positive"})

    Returns:
        RetrievalRawData with label texts as corpus

    Used by: Binary classification tasks when use_label_based=True
    """
    # Load dataset
    if hasattr(task, "hf_subset") and task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    # Get label field name
    label_field = getattr(task, "label_name", None) or getattr(task, "label", "label")

    # Get label texts
    if not hasattr(task, "label_texts"):
        raise ValueError(
            f"Task {task.__class__.__name__} must define 'label_texts' dict for label-based loading"
        )

    label_texts = task.label_texts

    # Build query-positive pairs
    query_ids = []
    positive_ids = []

    # Create unique query mapping
    unique_query_texts = []
    unique_query_ids = []
    text_to_query_id = {}

    for idx, row in enumerate(dataset):
        text = row[task.query_name]
        label = row[label_field]

        # Get label text
        if label not in label_texts:
            if rank == 0:
                print(f"Warning: Label {label} not in label_texts, skipping")
            continue

        positive_text = label_texts[label]

        # Create unique query ID if not seen
        if text not in text_to_query_id:
            query_id = f"query_{len(unique_query_ids)}"
            text_to_query_id[text] = query_id
            unique_query_ids.append(query_id)
            unique_query_texts.append(text)
        else:
            query_id = text_to_query_id[text]

        query_ids.append(query_id)
        positive_ids.append(f"label_{label}")

    # Build corpus: all label texts
    document_texts = list(label_texts.values())
    document_ids = [f"label_{label}" for label in label_texts.keys()]

    corpus_dict = {
        doc_id: {"text": doc_text}
        for doc_id, doc_text in zip(document_ids, document_texts)
    }

    # Build unique positives (all label texts)
    unique_positive_texts = list(label_texts.values())
    unique_positive_ids = [f"label_{label}" for label in label_texts.keys()]

    if rank == 0:
        print(f"Loaded {len(query_ids)} query-positive pairs for binary classification")
        print(f"Label texts: {label_texts}")

    return RetrievalRawData(
        query_ids=query_ids,
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
        documents_are_positives=False,
    )


def load_binary_classification_hard_negatives(task, rank=0):
    """
    Load binary classification datasets for hard negative mining.

    Similar to retrieval tasks, this loader creates a corpus of all texts,
    allowing for hard negative mining. Each text is a query, and texts with
    the same label are treated as positives for each other.

    Args:
        task: Task object with standard attributes

    Returns:
        RetrievalRawData with all texts as corpus for mining

    Used by: Binary classification tasks when use_hard_negative_mining=True
    """
    # Load dataset
    if hasattr(task, "hf_subset") and task.hf_subset:
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
        text = row[task.query_name]
        label = row[label_field]
        all_texts.append(text)
        all_labels.append(label)
        label_to_texts[label].append(text)

    # Build query-positive pairs
    # For each text, select a random text with the same label as positive
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

    for idx, (text, label) in enumerate(zip(all_texts, all_labels)):
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

    # Build corpus: all unique texts
    document_texts = list(text_to_id.keys())
    document_ids = list(text_to_id.values())

    corpus_dict = {
        doc_id: {"text": doc_text}
        for doc_id, doc_text in zip(document_ids, document_texts)
    }

    # Build unique positives
    unique_positive_texts = []
    unique_positive_ids = []
    seen_positive_ids = set()

    for pos_id in positive_ids:
        if pos_id not in seen_positive_ids:
            seen_positive_ids.add(pos_id)
            unique_positive_ids.append(pos_id)
            unique_positive_texts.append(corpus_dict[pos_id]["text"])

    if rank == 0:
        print(
            f"Loaded {len(query_ids)} query-positive pairs for binary classification with hard negative mining"
        )
        print(f"Corpus size: {len(document_texts)} unique texts")

    return RetrievalRawData(
        query_ids=query_ids,
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
        documents_are_positives=False,
    )
