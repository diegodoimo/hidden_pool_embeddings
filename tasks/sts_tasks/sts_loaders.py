"""
STS-specific loader functions.
These loaders handle Semantic Textual Similarity tasks.
"""

from datasets import load_dataset
from typing import List, Optional

from tasks.data_helpers import RetrievalRawData


def load_sts_retrieval(task) -> RetrievalRawData:
    """
    Load STS datasets for retrieval following Lee et al. (2025a).
    
    For textual similarity (STS) datasets, we construct a query-positive pair from any 
    sentence pair whose similarity score is at least 4 and another pair with the query 
    and positive switched. Hard negatives are mined from the corpus.
    
    Used by: STS12, STS22, STSBenchmark
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    score_name = getattr(task, "score_name", "score")
    score_threshold = getattr(task, "score_threshold", 4.0)

    query_texts = []
    positive_texts = []

    for row in dataset:
        score = row.get(score_name, 0)
        if score >= score_threshold:
            sentence1 = row[task.anchor_name]
            sentence2 = row[task.positive_name]
            
            # Create original pair: sentence1 as query, sentence2 as positive
            query_texts.append(sentence1)
            positive_texts.append(sentence2)
            
            # Create switched pair: sentence2 as query, sentence1 as positive
            query_texts.append(sentence2)
            positive_texts.append(sentence1)

    # Build corpus from all unique sentences (for hard negative mining)
    # Use a dict to deduplicate while preserving order
    text_to_id = {}
    doc_counter = 0
    
    # Add all texts (both queries and positives) to corpus for hard negative mining
    for text in query_texts + positive_texts:
        if text not in text_to_id:
            text_to_id[text] = f"doc_{doc_counter}"
            doc_counter += 1
    
    # Map positive texts to their corpus IDs
    positive_ids = [text_to_id[pos_text] for pos_text in positive_texts]
    
    # Generate query IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    
    # Build corpus
    document_texts = list(text_to_id.keys())
    document_ids = list(text_to_id.values())
    
    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # Get unique queries and positives
    unique_query_ids = []
    unique_query_texts = []
    seen_queries = set()
    
    for q_id, q_text in zip(query_ids, query_texts):
        if q_text not in seen_queries:
            seen_queries.add(q_text)
            unique_query_ids.append(q_id)
            unique_query_texts.append(q_text)

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
        unique_positive_texts=document_texts,
        unique_positive_ids=document_ids,
        unique_positive_titles=None,
        corpus_dict=corpus_dict,
        has_title=False,
    )
