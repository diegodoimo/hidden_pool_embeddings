"""
NLI-specific loader functions.
These loaders handle Natural Language Inference tasks.
"""

from datasets import load_dataset
from typing import List, Optional
import random
from collections import defaultdict

from tasks.data_helpers import RetrievalRawData


def load_nli_retrieval(task) -> RetrievalRawData:
    """
    Load NLI datasets (SNLI, MNLI, ANLI) for retrieval with hard negatives.
    
    For natural language inference (NLI) datasets, we retain only premises with at least one 
    entailed hypothesis as queries, and sample one of the entailed hypotheses as the positive. 
    If the query is also paired with neutral or contradictory hypotheses, we add them to the 
    corpus so they can be mined as hard negatives.
    
    Used by: SNLI, MNLI, ANLI, XNLI
    """
    if hasattr(task, "hf_subset") and task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)
    
    # Get label configuration
    entailment_label = getattr(task, "entailment_label", 0)
    neutral_label = getattr(task, "neutral_label", 1)
    contradiction_label = getattr(task, "contradiction_label", 2)
    
    # Group by premise to find all hypotheses for each premise
    premise_to_hypotheses = defaultdict(lambda: {"entailment": [], "non_entailment": []})
    
    for row in dataset:
        label = row[task.label_name]
        premise = row[task.anchor_name]
        hypothesis = row[task.positive_name]
        
        # Skip invalid labels (e.g., -1 in SNLI)
        if label < 0:
            continue
            
        if label == entailment_label:
            premise_to_hypotheses[premise]["entailment"].append(hypothesis)
        elif label in [neutral_label, contradiction_label]:
            # These will be added to corpus as potential hard negatives
            premise_to_hypotheses[premise]["non_entailment"].append(hypothesis)
    
    # Filter to keep only premises with at least one entailed hypothesis
    valid_premises = {
        premise: hyps 
        for premise, hyps in premise_to_hypotheses.items() 
        if len(hyps["entailment"]) > 0
    }
    
    # Build unique query ID mapping first (one-to-one: unique premise -> unique query ID)
    premise_to_query_id = {}
    unique_query_ids = []
    unique_query_texts = []
    
    for premise in valid_premises.keys():
        if premise not in premise_to_query_id:
            query_id = f"query_{len(unique_query_ids)}"
            premise_to_query_id[premise] = query_id
            unique_query_ids.append(query_id)
            unique_query_texts.append(premise)
    
    # Build the retrieval data (query-positive pairs)
    query_texts = []
    query_ids = []
    positive_texts = []
    positive_ids = []
    
    # Track unique documents for corpus
    hypothesis_to_id = {}
    doc_counter = 0
    
    for premise, hyps in valid_premises.items():
        # Sample one entailed hypothesis as positive
        positive_hypothesis = random.choice(hyps["entailment"])
        
        # Use the unique query ID for this premise
        query_id = premise_to_query_id[premise]
        
        # Get or create positive ID
        if positive_hypothesis not in hypothesis_to_id:
            hypothesis_to_id[positive_hypothesis] = f"doc_{doc_counter}"
            doc_counter += 1
        positive_id = hypothesis_to_id[positive_hypothesis]
        
        query_texts.append(premise)
        query_ids.append(query_id)
        positive_texts.append(positive_hypothesis)
        positive_ids.append(positive_id)
        
        # Add all non-entailment hypotheses to the corpus
        # These will be available as hard negatives during mining
        for neg_hyp in hyps["non_entailment"]:
            if neg_hyp not in hypothesis_to_id:
                hypothesis_to_id[neg_hyp] = f"doc_{doc_counter}"
                doc_counter += 1
    
    # Build corpus from all unique hypotheses (entailment + non-entailment)
    document_texts = []
    document_ids = []
    id_to_hypothesis = {v: k for k, v in hypothesis_to_id.items()}
    
    for doc_id in sorted(id_to_hypothesis.keys(), key=lambda x: int(x.split("_")[1])):
        document_ids.append(doc_id)
        document_texts.append(id_to_hypothesis[doc_id])
    
    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # Build unique positives (only from actual positives used in pairs, not all corpus)
    unique_positive_texts = []
    unique_positive_ids = []
    seen_positive_texts = set()
    
    for pos_text in positive_texts:
        if pos_text not in seen_positive_texts:
            seen_positive_texts.add(pos_text)
            unique_positive_texts.append(pos_text)
            unique_positive_ids.append(hypothesis_to_id[pos_text])
    
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


def load_all_nli_retrieval(task) -> RetrievalRawData:
    """
    Load ALL_NLI dataset which already has triplets (anchor, positive, negative).
    The negatives are included in the corpus so they can be mined as hard negatives.
    
    Used by: ALL_NLI
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    all_query_texts = list(dataset[task.anchor_name])
    all_positive_texts = list(dataset[task.positive_name])
    
    # Check if negative field exists
    has_negatives = hasattr(task, "negative_name") and task.negative_name in dataset.column_names
    if has_negatives:
        negative_texts = list(dataset[task.negative_name])
    else:
        negative_texts = []

    # Build unique query mapping (one-to-one: unique query text -> unique query ID)
    query_text_to_id = {}
    unique_query_texts = []
    unique_query_ids = []
    
    for query_text in all_query_texts:
        if query_text not in query_text_to_id:
            query_id = f"query_{len(unique_query_ids)}"
            query_text_to_id[query_text] = query_id
            unique_query_ids.append(query_id)
            unique_query_texts.append(query_text)

    # Build corpus: include both positives and negatives
    # Use a dict to deduplicate
    text_to_id = {}
    doc_counter = 0
    
    # Add positives
    for pos_text in all_positive_texts:
        if pos_text not in text_to_id:
            text_to_id[pos_text] = f"doc_{doc_counter}"
            doc_counter += 1
    
    # Add negatives to corpus
    if has_negatives:
        for neg_text in negative_texts:
            if neg_text not in text_to_id:
                text_to_id[neg_text] = f"doc_{doc_counter}"
                doc_counter += 1
    
    # Build query-positive pairs using unique query IDs
    query_texts = []
    query_ids = []
    positive_texts = []
    positive_ids = []
    
    for query_text, pos_text in zip(all_query_texts, all_positive_texts):
        query_texts.append(query_text)
        query_ids.append(query_text_to_id[query_text])
        positive_texts.append(pos_text)
        positive_ids.append(text_to_id[pos_text])
    
    # Build corpus
    document_ids = list(text_to_id.values())
    document_texts = list(text_to_id.keys())
    
    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # Build unique positives (only from actual positives, not negatives)
    unique_positive_texts = []
    unique_positive_ids = []
    seen_positive_texts = set()
    
    for pos_text in all_positive_texts:
        if pos_text not in seen_positive_texts:
            seen_positive_texts.add(pos_text)
            unique_positive_texts.append(pos_text)
            unique_positive_ids.append(text_to_id[pos_text])

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
