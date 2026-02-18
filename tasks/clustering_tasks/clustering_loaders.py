"""
Clustering-specific loader functions.
These loaders handle text clustering tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import random
from collections import defaultdict

from tasks.data_helpers import RetrievalRawData, ClassificationRawData


def load_clustering_standard(task, rank=0):
    """
    Standard loader for clustering tasks (legacy - simple format).
    Loads data from a single HuggingFace dataset with texts and labels.

    Returns ClassificationRawData format.
    Used by tasks that still need simple clustering format.
    """
    # Get label field name - could be "label", "label_name", or other custom field
    label_field = getattr(task, "label_name", None) or getattr(task, "label", "label")

    dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)

    texts = list(dataset[task.query_name])
    labels = list(dataset[label_field])
    ids = [f"doc_{i}" for i in range(len(texts))]

    if rank == 0:
        print(f"Loaded {len(texts)} samples for {task.metadata.type} task")

    return ClassificationRawData(texts=texts, labels=labels, ids=ids)


def load_clustering_sampling(task, rank=0, num_hard_negatives=24):
    """
    Load clustering datasets with in-batch sampling strategy.

    For each query:
    - A random sample from the same cluster is used as its positive passage
    - num_hard_negatives samples from other clusters are selected as hard negatives

    Args:
        task: Task object with standard attributes
        rank: Process rank for logging
        num_hard_negatives: Number of hard negatives from other clusters (default: 24)

    Returns:
        RetrievalRawData with sampled positives and negatives in corpus

    Used by: Clustering tasks
    """
    # Load dataset
    if hasattr(task, "hf_subset") and task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    # Get label field name (clusters are represented as labels)
    label_field = getattr(task, "label_name", None) or getattr(task, "label", "label")

    # Group texts by cluster/label
    cluster_to_texts = defaultdict(list)
    all_texts = []
    all_clusters = []

    for row in dataset:
        text = row[task.query_name]
        cluster = row[label_field]
        all_texts.append(text)
        all_clusters.append(cluster)
        cluster_to_texts[cluster].append(text)

    # Get all unique clusters
    unique_clusters = sorted(cluster_to_texts.keys())

    if rank == 0:
        print(f"Found {len(unique_clusters)} clusters in clustering task")
        for cluster in unique_clusters:
            print(f"  Cluster {cluster}: {len(cluster_to_texts[cluster])} samples")

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
    for text, cluster in zip(all_texts, all_clusters):
        # Get texts with same cluster
        same_cluster_texts = [t for t in cluster_to_texts[cluster] if t != text]

        # If there are other texts in same cluster, pick one as positive
        # Otherwise, use the text itself as positive
        if same_cluster_texts:
            positive_text = random.choice(same_cluster_texts)
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
    # During hard negative mining, the system will select hard negatives from other clusters
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
        print(f"Loaded {len(query_ids)} query-positive pairs for clustering")
        print(f"Corpus size: {len(document_texts)} unique texts")
        print(f"Hard negatives will be mined from corpus during training")

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


def load_clustering_hard_negatives(task, rank=0):
    """
    Load clustering datasets for hard negative mining.

    Similar to retrieval tasks, this loader creates a corpus of all texts,
    allowing for hard negative mining. Each text is a query, and texts in
    the same cluster are treated as positives for each other.

    This is an alias for the sampling version since they both support hard negative mining.

    Args:
        task: Task object with standard attributes
        rank: Process rank for logging

    Returns:
        RetrievalRawData with all texts as corpus for mining

    Used by: Clustering tasks when use_hard_negative_mining=True
    """
    return load_clustering_sampling(task, rank=rank)
