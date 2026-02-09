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

    # First pass: collect all sentence pairs and build corpus
    all_sentence_pairs = []
    all_texts_set = set()
    
    for row in dataset:
        score = row.get(score_name, 0)
        if score >= score_threshold:
            sentence1 = row[task.anchor_name]
            sentence2 = row[task.positive_name]
            
            all_texts_set.add(sentence1)
            all_texts_set.add(sentence2)
            
            # Create original pair: sentence1 as query, sentence2 as positive
            all_sentence_pairs.append((sentence1, sentence2))
            
            # Only create switched pair if sentences are different
            if sentence1 != sentence2:
                all_sentence_pairs.append((sentence2, sentence1))
    
    # Build corpus from all unique sentences and assign them doc IDs
    text_to_doc_id = {}
    document_texts = []
    document_ids = []
    
    for idx, text in enumerate(sorted(all_texts_set)):  # Sort for reproducibility
        doc_id = f"doc_{idx}"
        text_to_doc_id[text] = doc_id
        document_texts.append(text)
        document_ids.append(doc_id)
    
    corpus_dict = {
        doc_id: {"text": text} for doc_id, text in zip(document_ids, document_texts)
    }
    
    # Build query text to unique query ID mapping
    # This ensures one-to-one mapping between unique query texts and IDs
    query_text_to_id = {}
    unique_query_texts = []
    unique_query_ids = []
    
    for query_text, _ in all_sentence_pairs:
        if query_text not in query_text_to_id:
            query_id = f"query_{len(unique_query_ids)}"
            query_text_to_id[query_text] = query_id
            unique_query_ids.append(query_id)
            unique_query_texts.append(query_text)
    
    # Build full query-positive pairs using the mapped query IDs
    query_ids = []
    query_texts = []
    positive_ids = []
    positive_texts = []
    
    for query_text, positive_text in all_sentence_pairs:
        # Use the unique query ID for this query text
        query_ids.append(query_text_to_id[query_text])
        query_texts.append(query_text)
        
        # Use the corpus doc ID for the positive
        positive_ids.append(text_to_doc_id[positive_text])
        positive_texts.append(positive_text)
    
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
