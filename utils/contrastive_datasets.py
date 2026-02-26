import torch
import numpy as np
import os
import glob
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import Dataset, concatenate_datasets
from mteb.types import PromptType

from torch.utils.data import DistributedSampler

from functools import partial

from inference.create_datasets import (
    _build_prompt,
    _remove_long_sequences_hard_negatives,
)

# taken from embeddinggemma
# https://github.com/huggingface/transformers/blob/bdee0889714e9cb3e53d3b1b2a626919479d356c/src/transformers/models/gemma3/convert_gemma3_weights.py#L700C1-L715C10
TASK_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: {title} | text: {text}",
    "BitextMining": "task: search result | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: ",
}


# ---------------------------------------------------------------------------
# Hard negatives dataset loading and tokenization
# ---------------------------------------------------------------------------


@dataclass
class TrainTaskMetadata:
    type: str
    prompt: str = None


FOLDER_TO_TASK = {
    "retrieval/general_retrieval": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a web search query, retrieve relevant passages that answer the query",
    ),
    "retrieval/domain_specific_qa": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a question, retrieve passages that answer the question",
    ),
    "retrieval/open_domain_qa": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a question, retrieve passages that answer the question",
    ),
    "retrieval/fact_verification": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a claim, retrieve documents that support or refute the claim",
    ),
    "retrieval/paraphrase_detection": TrainTaskMetadata(
        type="STS",
        prompt="Retrieve semantically similar text",
    ),
    "retrieval/scientific_doc_retrieval": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a scientific paper title, retrieve the paper abstract",
    ),
    "retrieval/summarization": TrainTaskMetadata(
        type="Summarization",
        prompt="Given a summary, retrieve the original document",
    ),
    "nli": TrainTaskMetadata(
        type="Classification",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
    ),
    "sts": TrainTaskMetadata(
        type="STS",
        prompt="Retrieve semantically similar text",
    ),
}

DEFAULT_TASK = TrainTaskMetadata(
    type="Retrieval",
    prompt="Given a web search query, retrieve relevant passages that answer the query",
)


def _infer_task_metadata(parquet_path, base_dir):
    """Infer task metadata from the directory structure."""
    rel = os.path.relpath(os.path.dirname(parquet_path), base_dir)
    parts = rel.replace(os.sep, "/").split("/")
    for depth in range(len(parts), 0, -1):
        key = "/".join(parts[:depth])
        if key in FOLDER_TO_TASK:
            return FOLDER_TO_TASK[key]
    return DEFAULT_TASK


def _load_parquet_safe(path):
    """Load a parquet file as an HF Dataset with fallback for metadata issues."""
    try:
        return Dataset.from_parquet(path)
    except (TypeError, Exception):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        table = table.replace_schema_metadata({})
        return Dataset(table)


def _str_to_int_id(s: str) -> int:
    """Deterministic hash of a string to a positive 63-bit integer."""
    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)


def _build_prompts_hard_negatives_batch(
    examples,
    tokenizer,
    instruction_template,
    task_metadata,
    num_hard_negatives,
):
    """Build prompts for (query, positive, negatives) using create_datasets._build_prompt.

    Mirrors the create_dataset flow: uses _build_prompt for query, positive, and negatives.
    """
    batch_size = len(examples["query_text"])
    eot_id = tokenizer.pad_token_id

    # Query prompts (mirror create_dataset: map with _build_prompt)
    q_rows = {
        "text": examples["query_text"],
        "id": examples.get("query_id", [str(i) for i in range(batch_size)]),
    }
    query_result = _build_prompt(
        q_rows, tokenizer, instruction_template, PromptType.query, task_metadata, eot_id
    )
    query_prompts = query_result["prompt"]

    # Positive prompts
    pos_titles = examples.get("positive_title", None)
    if pos_titles is None:
        pos_titles = [""] * batch_size
    p_rows = {
        "text": examples["positive_text"],
        "id": examples["positive_id"],
        "title": pos_titles,
    }
    pos_result = _build_prompt(
        p_rows,
        tokenizer,
        instruction_template,
        PromptType.document,
        task_metadata,
        eot_id,
    )
    positive_prompts = pos_result["prompt"]

    # Negative prompts: flatten, build, unflatten
    all_neg_texts = []
    all_neg_ids = []
    all_neg_titles = []
    for i in range(batch_size):
        neg_texts = examples["negative_text"][i][:num_hard_negatives]
        neg_titles_col = examples.get("negative_title", None)
        if neg_titles_col and neg_titles_col[i]:
            neg_titles = neg_titles_col[i][:num_hard_negatives]
        else:
            neg_titles = [""] * len(neg_texts)
        all_neg_texts.extend(neg_texts)
        all_neg_ids.extend([f"{i}_{j}" for j in range(len(neg_texts))])
        all_neg_titles.extend(neg_titles)

    if all_neg_texts:
        n_rows = {"text": all_neg_texts, "id": all_neg_ids, "title": all_neg_titles}
        neg_result = _build_prompt(
            n_rows,
            tokenizer,
            instruction_template,
            PromptType.document,
            task_metadata,
            eot_id,
        )
        neg_prompts_flat = neg_result["prompt"]
        # Unflatten
        idx = 0
        negative_prompts = []
        for i in range(batch_size):
            n = min(len(examples["negative_text"][i]), num_hard_negatives)
            negative_prompts.append(neg_prompts_flat[idx : idx + n])
            idx += n
    else:
        negative_prompts = [[] for _ in range(batch_size)]

    return {
        "query_prompt": query_prompts,
        "positive_prompt": positive_prompts,
        "negative_prompts": negative_prompts,
    }


def _add_lengths_and_dataset_name(
    examples,
    tokenizer,
    max_query_len,
    max_passage_len,
    num_hard_negatives,
    dataset_name,
):
    """Add dataset_name and token lengths for sorting. Keeps prompts for collate tokenization."""
    batch_size = len(examples["query_prompt"])

    query_encs = tokenizer(
        examples["query_prompt"],
        max_length=max_query_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )
    pos_encs = tokenizer(
        examples["positive_prompt"],
        max_length=max_passage_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    all_avg_neg_len = []
    for i in range(batch_size):
        neg_prompts = examples["negative_prompts"][i][:num_hard_negatives]
        if neg_prompts:
            neg_encs = tokenizer(
                neg_prompts,
                max_length=max_passage_len,
                truncation=True,
                padding=False,
                return_attention_mask=False,
            )
            avg_len = np.mean([len(n) for n in neg_encs["input_ids"]])
        else:
            avg_len = len(pos_encs["input_ids"][i])
        all_avg_neg_len.append(avg_len)

    return {
        "dataset_name": [dataset_name] * batch_size,
        "query_len": [len(ids) for ids in query_encs["input_ids"]],
        "pos_len": [len(ids) for ids in pos_encs["input_ids"]],
        "total_len": [
            len(query_encs["input_ids"][i])
            + len(pos_encs["input_ids"][i])
            + int(all_avg_neg_len[i] * num_hard_negatives)
            for i in range(batch_size)
        ],
    }


# Example subset of 10 datasets from results/datasets_negatives/qwen3_600m leaf folders.
# Use as datasets_subset=QWEN3_600M_10DATASET_SUBSET to restrict training to these.
QWEN3_600M_DATASET_SUBSET = [
    # "retrieval/general_retrieval/msmarco",
    "retrieval/general_retrieval/nfcorpus",
    "retrieval/general_retrieval/arguana",
    "retrieval/domain_specific_qa/fiqa2018",
    "retrieval/open_domain_qa/naturalquestions",
    "retrieval/open_domain_qa/squad",
    "retrieval/fact_verification/scifact",
    "retrieval/summarization/xsum",
    "sts/stsbenchmark",
    "nli/snli",
]


def load_hard_negatives_datasets(
    base_dir,
    num_hard_negatives,
    tokenizer,
    instruction_template,
    max_query_len=256,
    max_passage_len=512,
    rank=0,
    datasets_subset: Optional[List[str]] = None,
):
    """Load and tokenize all hard-negative parquet datasets under *base_dir*.

    Mirrors the logic of create_datasets.create_dataset:
    1. Map: build prompts using _build_prompt (via _build_prompts_hard_negatives_batch)
    2. Filter: remove long sequences using _remove_long_sequences_hard_negatives
    3. Map: tokenize the prompts

    Args:
        base_dir: Root directory containing dataset subdirs with data.parquet
        num_hard_negatives: Number of hard negatives per example
        tokenizer: HuggingFace tokenizer
        instruction_template: Callable for building instruction prompts
        max_query_len: Max query token length
        max_passage_len: Max passage token length
        rank: Process rank (0 = main, for logging)
        datasets_subset: Optional list of dataset names (relative paths from base_dir)
            to restrict loading. Names should match leaf folders, e.g.
            "retrieval/general_retrieval/msmarco", "sts/stsbenchmark".
            Use QWEN3_600M_10DATASET_SUBSET for a 10-dataset example.

    Returns a single concatenated HF Dataset sorted by total sequence length
    (longest first) for length-balanced batching.
    """

    parquet_files = sorted(
        glob.glob(os.path.join(base_dir, "**", "data.parquet"), recursive=True)
    )

    if datasets_subset is not None:
        subset_set = set(datasets_subset)
        parquet_files = [
            p
            for p in parquet_files
            if os.path.relpath(os.path.dirname(p), base_dir) in subset_set
        ]
        if rank == 0:
            print(
                f"Restricted to {len(parquet_files)} datasets (subset of {len(datasets_subset)} requested)"
            )

    if rank == 0:
        print(f"Found {len(parquet_files)} datasets under {base_dir}")

    all_datasets = []
    for path in parquet_files:
        task_metadata = _infer_task_metadata(path, base_dir)
        ds_name = os.path.relpath(os.path.dirname(path), base_dir)

        if rank == 0:
            print(f"  Loading {ds_name} ...")

        ds = _load_parquet_safe(path)

        # Keep only examples with at least num_hard_negatives negatives
        ds = ds.filter(
            lambda x: len(x["negative_text"]) >= num_hard_negatives,
            num_proc=1,
        )

        if rank == 0:
            print(
                f"    {len(ds)} examples after filtering (>= {num_hard_negatives} negatives)"
            )

        if len(ds) == 0:
            continue

        # Step 1: Build prompts (mirror create_dataset: map with _build_prompt)
        start = time.time()
        build_fn = partial(
            _build_prompts_hard_negatives_batch,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            task_metadata=task_metadata,
            num_hard_negatives=num_hard_negatives,
        )
        ds = ds.map(build_fn, batched=True, batch_size=10000)
        if rank == 0:
            print(f"    prompt constructed in {(time.time()-start)/60:.2f}min")

        # Step 2: Filter long sequences (mirror create_dataset: filter with _remove_long_sequences)
        all_removed_long_ids = []
        all_removed_empty_ids = []

        def filter_wrapper(rows):
            keep_mask, removed_long, removed_empty = _remove_long_sequences_hard_negatives(
                rows, tokenizer, max_query_len, max_passage_len
            )
            all_removed_long_ids.extend(removed_long)
            all_removed_empty_ids.extend(removed_empty)
            return keep_mask

        start = time.time()
        ds = ds.filter(filter_wrapper, batched=True, batch_size=10000)
        if rank == 0:
            print(f"    dataset filtered in {(time.time()-start)/60:.2f}min")

        if len(ds) == 0:
            continue

        # Step 3: Add lengths and dataset_name for sorting (tokenization done in collate)
        len_fn = partial(
            _add_lengths_and_dataset_name,
            tokenizer=tokenizer,
            max_query_len=max_query_len,
            max_passage_len=max_passage_len,
            num_hard_negatives=num_hard_negatives,
            dataset_name=ds_name,
        )
        ds = ds.map(len_fn, batched=True, batch_size=1000)
        # Keep only columns needed for collate
        cols_to_keep = {
            "query_prompt",
            "positive_prompt",
            "negative_prompts",
            "positive_id",
            "dataset_name",
            "query_len",
            "pos_len",
            "total_len",
        }
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        ds = ds.remove_columns(cols_to_remove)
        all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)
    combined = combined.sort("total_len", reverse=True)

    if rank == 0:
        total_tokens = np.sum(combined["total_len"])
        print(f"Total training examples: {len(combined)/1e3:.1f}k")
        print(f"Total tokens: {total_tokens/1e6:.1f}M")
        print(f"Avg query len: {np.mean(combined['query_len']):.0f}")
        print(f"Avg doc len: {np.mean(combined['pos_len']):.0f}")

    return combined
