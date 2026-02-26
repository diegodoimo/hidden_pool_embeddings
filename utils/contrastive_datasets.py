import torch
import numpy as np
import os
import glob
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import Dataset, concatenate_datasets
from mteb.types import PromptType

from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from functools import partial

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


class LengthBalancedDistributedSampler(DistributedSampler):
    """
    Distributed sampler that ensures each GPU gets batches with balanced lengths.

    Strategy: Sort by length, then interleave across GPUs so each gets a mix of short/long examples.
    """

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self):
        if self.shuffle:
            # Shuffle within buckets to add randomness while maintaining balance
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Create bucket-shuffled indices
            bucket_size = self.num_replicas * 100  # Each bucket = 100 batches per GPU
            indices = []

            for start in range(0, len(self.dataset), bucket_size):
                end = min(start + bucket_size, len(self.dataset))
                bucket = list(range(start, end))
                # Shuffle within bucket
                bucket_indices = torch.tensor(bucket)[
                    torch.randperm(len(bucket), generator=g)
                ].tolist()
                indices.extend(bucket_indices)
        else:
            indices = list(range(len(self.dataset)))

        # Distribute indices round-robin to ensure balanced lengths across GPUs
        # GPU 0 gets: 0, num_replicas, 2*num_replicas, ...
        # GPU 1 gets: 1, num_replicas+1, 2*num_replicas+1, ...
        indices = indices[self.rank :: self.num_replicas]

        # Pad if needed
        if not self.drop_last:
            padding_size = self.num_samples - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.num_samples]

        return iter(indices)


def collate_fn_with_padding(batch, pad_token_id=0):
    """
    Collate function that pads sequences and creates attention masks.

    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID used for padding (usually 0)

    Returns:
        Dict with padded input_ids and attention_masks
    """
    query_token_ids = [torch.tensor(item["query_token_ids"]) for item in batch]
    pos_token_ids = [torch.tensor(item["pos_token_ids"]) for item in batch]
    pos_ids = torch.cat([torch.tensor([item["pos_ids"]]) for item in batch])

    # Handle neg_token_ids (list of lists)

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids, batch_first=True, padding_value=pad_token_id
    )
    query_attention_mask = (query_token_ids_padded != pad_token_id).long()

    # Pad positive passages and create attention masks
    pos_token_ids_padded = pad_sequence(pos_token_ids, batch_first=True, padding_value=pad_token_id)
    pos_attention_mask = (pos_token_ids_padded != pad_token_id).long()

    return {
        "query_token_ids": query_token_ids_padded,
        "query_attention_mask": query_attention_mask,
        "pos_token_ids": pos_token_ids_padded,
        "pos_attention_mask": pos_attention_mask,
        "pos_ids": pos_ids,
    }


def collate_fn_with_padding_joint(batch, pad_token_id=0):
    """
    Collate function that pads sequences and creates attention masks.

    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID used for padding (usually 0)

    Returns:
        Dict with padded input_ids and attention_masks
    """

    inputs_token_ids = [torch.tensor(item["query_token_ids"]) for item in batch]
    inputs_token_ids.extend([torch.tensor(item["pos_token_ids"]) for item in batch])
    pos_ids = torch.cat([torch.tensor([item["pos_ids"]]) for item in batch])

    # Pad inputs and create attention masks
    inputs_token_ids_padded = pad_sequence(
        inputs_token_ids, batch_first=True, padding_value=pad_token_id
    )
    inputs_attention_mask = (inputs_token_ids_padded != pad_token_id).long()

    return {
        "query_token_ids": inputs_token_ids,
        "query_attention_mask": inputs_attention_mask,
        "pos_ids": pos_ids,
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


def tokenize_hard_negatives_batch(
    examples,
    tokenizer,
    instruction_template,
    task_metadata,
    max_query_len,
    max_passage_len,
    num_hard_negatives,
    dataset_name,
):
    """Tokenize a batch of (query, positive, negatives) triples.

    Mirrors the prompt-building logic from create_datasets._build_prompt
    and the tokenization from contrastive_datasets.tokenize_batch.
    """
    batch_size = len(examples["query_text"])

    # --- Build prompts ---
    query_prompts = [
        instruction_template(PromptType.query, task_metadata, text)
        for text in examples["query_text"]
    ]

    pos_titles = examples.get("positive_title", None)
    if pos_titles is None:
        pos_titles = [""] * batch_size

    pos_prompts = [
        instruction_template(
            PromptType.document, task_metadata, text, title=title or ""
        )
        for text, title in zip(examples["positive_text"], pos_titles)
    ]

    # --- Tokenize queries and positives (batch) ---
    query_encs = tokenizer(
        query_prompts,
        max_length=max_query_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )
    pos_encs = tokenizer(
        pos_prompts,
        max_length=max_passage_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    # --- Process negatives per example ---
    neg_titles_col = examples.get("negative_title", None)
    all_neg_token_ids = []
    all_avg_neg_len = []

    for i in range(batch_size):
        neg_texts = examples["negative_text"][i][:num_hard_negatives]

        if neg_titles_col and neg_titles_col[i]:
            neg_titles = neg_titles_col[i][:num_hard_negatives]
        else:
            neg_titles = [""] * len(neg_texts)

        neg_prompts = [
            instruction_template(
                PromptType.document, task_metadata, text, title=title or ""
            )
            for text, title in zip(neg_texts, neg_titles)
        ]

        neg_encs = tokenizer(
            neg_prompts,
            max_length=max_passage_len,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )
        neg_ids_list = neg_encs["input_ids"]

        # Pad with the positive if not enough negatives
        while len(neg_ids_list) < num_hard_negatives:
            neg_ids_list.append(pos_encs["input_ids"][i])

        neg_ids_list = neg_ids_list[:num_hard_negatives]
        all_neg_token_ids.append(neg_ids_list)
        all_avg_neg_len.append(np.mean([len(n) for n in neg_ids_list]))

    # --- Assemble result ---
    pos_ids = [
        _str_to_int_id(f"{dataset_name}/{pid}")
        for pid in examples["positive_id"]
    ]

    result = {
        "query_token_ids": query_encs["input_ids"],
        "pos_token_ids": pos_encs["input_ids"],
        "neg_token_ids": all_neg_token_ids,
        "pos_ids": pos_ids,
        "query_len": [len(ids) for ids in query_encs["input_ids"]],
        "pos_len": [len(ids) for ids in pos_encs["input_ids"]],
        "total_len": [
            len(query_encs["input_ids"][i])
            + len(pos_encs["input_ids"][i])
            + int(all_avg_neg_len[i] * num_hard_negatives)
            for i in range(batch_size)
        ],
    }
    return result


# Example subset of 10 datasets from results/datasets_negatives/qwen3_600m leaf folders.
# Use as datasets_subset=QWEN3_600M_10DATASET_SUBSET to restrict training to these.
QWEN3_600M_DATASET_SUBSET = [
    #"retrieval/general_retrieval/msmarco",
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
            p for p in parquet_files
            if os.path.relpath(os.path.dirname(p), base_dir) in subset_set
        ]
        if rank == 0:
            print(f"Restricted to {len(parquet_files)} datasets (subset of {len(datasets_subset)} requested)")

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
            print(f"    {len(ds)} examples after filtering (>= {num_hard_negatives} negatives)")

        if len(ds) == 0:
            continue

        tok_fn = partial(
            tokenize_hard_negatives_batch,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            task_metadata=task_metadata,
            max_query_len=max_query_len,
            max_passage_len=max_passage_len,
            num_hard_negatives=num_hard_negatives,
            dataset_name=ds_name,
        )

        tokenized = ds.map(
            tok_fn,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
        )
        all_datasets.append(tokenized)

    combined = concatenate_datasets(all_datasets)
    combined = combined.sort("total_len", reverse=True)

    if rank == 0:
        total_tokens = np.sum(combined["total_len"])
        print(f"Total training examples: {len(combined)/1e3:.1f}k")
        print(f"Total tokens: {total_tokens/1e6:.1f}M")
        print(f"Avg query len: {np.mean(combined['query_len']):.0f}")
        print(f"Avg doc len: {np.mean(combined['pos_len']):.0f}")

    return combined


def collate_fn_with_hard_negatives(batch, pad_token_id=0, num_hard_negatives=8):
    """Collate function for batches that include hard negatives.

    Returns padded tensors for queries, positives, and negatives with their
    attention masks.
    """
    query_token_ids = [torch.tensor(item["query_token_ids"]) for item in batch]
    pos_token_ids = [torch.tensor(item["pos_token_ids"]) for item in batch]
    pos_ids = torch.tensor([item["pos_ids"] for item in batch], dtype=torch.long)

    # Pad queries
    query_padded = pad_sequence(
        query_token_ids, batch_first=True, padding_value=pad_token_id
    )
    query_mask = (query_padded != pad_token_id).long()

    # Pad positives
    pos_padded = pad_sequence(
        pos_token_ids, batch_first=True, padding_value=pad_token_id
    )
    pos_mask = (pos_padded != pad_token_id).long()

    # Flatten all negatives across the batch, pad, then reshape
    all_neg_seqs = []
    for item in batch:
        for neg in item["neg_token_ids"]:
            all_neg_seqs.append(torch.tensor(neg))

    neg_padded = pad_sequence(
        all_neg_seqs, batch_first=True, padding_value=pad_token_id
    )
    neg_mask = (neg_padded != pad_token_id).long()

    batch_size = len(batch)
    neg_seq_len = neg_padded.size(1)
    neg_padded = neg_padded.view(batch_size, num_hard_negatives, neg_seq_len)
    neg_mask = neg_mask.view(batch_size, num_hard_negatives, neg_seq_len)

    return {
        "query_token_ids": query_padded,
        "query_attention_mask": query_mask,
        "pos_token_ids": pos_padded,
        "pos_attention_mask": pos_mask,
        "neg_token_ids": neg_padded,
        "neg_attention_mask": neg_mask,
        "pos_ids": pos_ids,
    }
