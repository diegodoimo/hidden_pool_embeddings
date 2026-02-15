from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries


def load_squad_retrieval(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """Load SQuAD dataset for retrieval using vectorized operations.

    Question is used as query, context is used as positive document.
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start = time.time()
        print("Loading dataset...")
    
    dataset = load_dataset(task.hf_name, split=task.split)
    
    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(f"num elements in dataset: {return_formatted(len(dataset))}")
        start = time.time()
        print("Converting to pandas...")

    # Convert to pandas DataFrame
    df = dataset.select_columns([task.anchor_name, task.positive_name]).to_pandas()
    df.columns = ["question", "context"]
    n_pairs = len(df)
    
    dist.barrier()
    if rank == 0:
        print(f"Conversion done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Building query-positive pairs with deduplication...")
    
    # Get unique queries (questions don't need deduplication - each question is unique)
    unique_query_mask = ~df["question"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_texts = df.loc[unique_query_mask, "question"].tolist()
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]
    
    # Create query text to ID mapping
    query_text_to_id = pd.Series(unique_query_ids, index=df.loc[unique_query_mask, "question"].values)
    
    # Map all queries to their unique IDs
    query_ids = df["question"].map(query_text_to_id).tolist()
    
    # Get unique contexts (positives) - contexts are often repeated across questions
    unique_positive_mask = ~df["context"].duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index.values
    unique_positive_texts = df.loc[unique_positive_mask, "context"].tolist()
    unique_positive_ids = [f"doc_{i}" for i in range(len(unique_positive_texts))]
    
    # Create context text to ID mapping
    context_text_to_id = pd.Series(unique_positive_ids, index=df.loc[unique_positive_mask, "context"].values)
    
    # Map all contexts to their unique IDs
    positive_ids = df["context"].map(context_text_to_id).tolist()
    
    dist.barrier()
    if rank == 0:
        print(f"Deduplication done in {(time.time()-start)/60:.2f} min")
        print(f"Found {return_formatted(len(unique_query_ids))} unique queries before limiting")
        print(f"Found {return_formatted(len(unique_positive_ids))} unique contexts")
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
        # In SQuAD, all documents are positives (no extra corpus documents)
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


class SQuAD(AbsTask):
    """SQuAD dataset for retrieval - question as query, context as positive."""

    language = "en"

    hf_name = "rajpurkar/squad"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )
    loader = load_squad_retrieval
