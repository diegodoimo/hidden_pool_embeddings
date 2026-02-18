from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


def load_pubmedqa_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """Load PubMedQA dataset for retrieval using vectorized operations.

    Question is query, context (long_answer) is positive document.
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset...")
    
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)
    
    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(len(dataset))}")
        start = time.time()
        print("Extracting questions and contexts...")

    # Extract questions and contexts (handling various data formats)
    query_texts = []
    positive_texts = []

    for row in dataset:
        question = row.get("question", "")
        # Context can be a list or dict with 'contexts' field
        context = row.get("context", {})
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if contexts:
                positive_text = " ".join(contexts)
            else:
                continue
        elif isinstance(context, list):
            positive_text = " ".join(context)
        elif isinstance(context, str):
            positive_text = context
        else:
            continue

        if question and positive_text:
            query_texts.append(question)
            positive_texts.append(positive_text)

    # Convert to pandas DataFrame
    df = pd.DataFrame({"question": query_texts, "context": positive_texts})
    n_pairs = len(df)
    
    dist.barrier()
    if rank == 0:
        print(f"Extraction done in {(time.time()-start)/60:.2f} min")
        print(f"Valid pairs extracted: {return_formatted(n_pairs)}")
        start = time.time()
        print("Building query-positive pairs with deduplication...")
    
    # Get unique queries using pandas
    unique_query_mask = ~df["question"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_texts = df.loc[unique_query_mask, "question"].tolist()
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]
    
    # Create query text to ID mapping
    query_text_to_id = pd.Series(unique_query_ids, index=df.loc[unique_query_mask, "question"].values)
    
    # Map all queries to their unique IDs
    query_ids = df["question"].map(query_text_to_id).tolist()
    
    # Get unique positives
    unique_positive_mask = ~df["context"].duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index.values
    unique_positive_texts = df.loc[unique_positive_mask, "context"].tolist()
    unique_positive_ids = [f"doc_{i}" for i in unique_positive_idx]
    
    # Create document text to ID mapping
    doc_text_to_id = pd.Series(unique_positive_ids, index=df.loc[unique_positive_mask, "context"].values)
    
    # Map all positives to their unique IDs
    positive_ids = df["context"].map(doc_text_to_id).tolist()
    
    dist.barrier()
    if rank == 0:
        print(f"Deduplication done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(unique_query_ids))} unique queries before limiting")
        print(f"Found {return_formatted(len(unique_positive_ids))} unique documents")
        start = time.time()
        print("Applying query limiting...")
    
    # Apply query limiting if needed
    if max_num_queries is not None and len(unique_query_idx) > max_num_queries:
        if rank == 0:
            print(
                f"Number of unique queries {return_formatted(len(unique_query_idx))} > {max_num_queries//10**6}M: limiting queries"
            )
        
        unique_query_texts = unique_query_texts[:max_num_queries]
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_idx = unique_query_idx[:max_num_queries]
        
        (
            query_ids,
            positive_ids,
            document_ids,
            document_texts,
            _,
            n_positives,
        ) = limit_number_of_queries(
            query_ids=query_ids,
            positive_ids=positive_ids,
            unique_query_idx=unique_query_idx,
            n_pairs=n_pairs,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=None,
            has_title=False,
            max_queries=max_num_queries,
        )
        
        if rank == 0:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
            print(f"Positives referenced by filtered pairs: {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")
    else:
        # No limiting needed
        document_ids = unique_positive_ids
        document_texts = unique_positive_texts
        n_positives = len(unique_positive_ids)
        
        if rank == 0:
            print(
                f"Found {return_formatted(len(unique_query_ids))} unique queries"
            )
            print(f"Total number of query-positive pairs: {return_formatted(len(query_ids))}")
            print(f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}")
            print(f"Total unique documents in corpus: {return_formatted(len(document_ids))}")

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # Assertions to ensure data consistency
    assert set(positive_ids).issubset(
        set(document_ids)
    ), "filtered qrels contain positive IDs not in document list"
    
    assert set(unique_positive_ids).issubset(
        set(document_ids)
    ), "unique positives not in document list"
    
    assert set(corpus_dict.keys()) == set(document_ids), "corpus_dict keys mismatch with document_ids"

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=False,
        n_positives=n_positives,
    )


class PubMedQA(AbsTask):
    """PubMedQA biomedical QA dataset for retrieval."""

    language = "en"

    hf_name = "qiaojin/PubMedQA"
    hf_subset = "pqa_labeled"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )
    loader = load_pubmedqa_retrieval
