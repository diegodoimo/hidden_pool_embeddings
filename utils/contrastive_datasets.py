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


def prepare_msmarco(hf_queries, hf_corpus, hf_qrels):
    query_to_doc_map = {
        example["query-id"]: example["corpus-id"] for example in hf_qrels if example["score"] == 1
    }

    # Get only the IDs we need
    relevant_doc_ids = set(query_to_doc_map.values())
    relevant_query_token_ids = set(query_to_doc_map.keys())

    # Filter while loading (still batched for speed)
    def build_filtered_dict(dataset, relevant_ids, batch_size=10000):
        filtered_dict = {}
        for batch in dataset.iter(batch_size=batch_size):
            for i in range(len(batch["_id"])):
                doc_id = batch["_id"][i]
                if doc_id in relevant_ids:
                    filtered_dict[doc_id] = {
                        "_id": doc_id,
                        "title": batch["title"][i] if "title" in batch else None,
                        "text": batch["text"][i],
                    }
        return filtered_dict

    corpus_full = build_filtered_dict(hf_corpus["corpus"], relevant_doc_ids)
    queries_full = {item["_id"]: item for item in hf_queries["queries"]}

    # Build aligned data
    data_query = {"query_id": [], "query_text": [], "positive_id": []}

    data_doc = {"positive_id": [], "positive_title": [], "positive_text": []}

    for query_id, doc_id in query_to_doc_map.items():
        if query_id in queries_full and doc_id in corpus_full:
            data_query["query_id"].append(query_id)
            data_query["query_text"].append(queries_full[query_id]["text"])
            data_query["positive_id"].append(doc_id)

            data_doc["positive_id"].append(doc_id)
            title = corpus_full[doc_id]["title"]
            data_doc["positive_title"].append(title if len(title) > 0 else "none")
            data_doc["positive_text"].append(corpus_full[doc_id]["text"])

    # Create HuggingFace dataset
    # train_queries = Dataset.from_dict(data_query)
    # train_docs = Dataset.from_dict(data_doc)
    return data_query, data_doc


def tokenize_batch(
    examples,
    query_prompt,
    doc_prompt,
    tokenizer,
    max_query_len,
    max_passage_len,
    num_hard_negatives=None,
) -> Dict[str, List]:
    """Tokenize a batch of examples with prompts."""
    batch_size = len(examples["query"])

    # Prepend prompts
    queries_with_prompt = [query_prompt + q for q in examples["query"]]
    pos_with_prompt = [
        doc_prompt.format(title=title, text=text)
        for title, text in zip(examples["pos_title"], examples["pos_passage"])
    ]

    # Tokenize queries
    query_encs = tokenizer(
        queries_with_prompt,
        max_length=max_query_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    # Tokenize positive passages
    pos_encs = tokenizer(
        pos_with_prompt,
        max_length=max_passage_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    # Initialize output
    result = {
        "query_text": queries_with_prompt,
        "pos_text": pos_with_prompt,
        "query_token_ids": query_encs["input_ids"],
        "pos_token_ids": pos_encs["input_ids"],
        "pos_ids": [int(ids) for ids in examples["pos_ids"]],
        "query_len": [len(ids) for ids in query_encs["input_ids"]],
        "pos_len": [len(ids) for ids in pos_encs["input_ids"]],
        "total_len": [],
    }

    # Process negatives if available
    if "neg_passages" in examples:
        for i in range(batch_size):
            neg_passages = examples["neg_passages"][i]

            # Take first num_hard_negatives
            neg_passages_subset = neg_passages[:num_hard_negatives]

            # Prepend document prompt to negatives
            neg_with_prompt = [doc_prompt + neg for neg in neg_passages_subset]

            # Tokenize negatives
            neg_encs = tokenizer(
                neg_with_prompt,
                max_length=max_passage_len,
                truncation=True,
                padding=False,
                return_attention_mask=False,
            )

            neg_token_ids_list = neg_encs["input_ids"]

            # Pad with positive passage if not enough negatives
            # while len(neg_token_ids_list) < num_hard_negatives:
            #     neg_token_ids_list.append(pos_encs["input_ids"][i])

            result["neg_token_ids"].append(neg_token_ids_list)

            # Calculate average negative length
            avg_neg_len = np.mean([len(neg) for neg in neg_token_ids_list])
            result["avg_neg_len"].append(avg_neg_len)
            result["total_len"].append(
                result["query_len"][i] + result["pos_len"][i] + avg_neg_len * num_hard_negatives
            )
    else:
        # No negatives provided
        for i in range(batch_size):
            # result["neg_token_ids"].append([pos_encs["input_ids"][i]] * num_hard_negatives)
            # result["avg_neg_len"].append(result["pos_len"][i])
            # result["total_len"].append(
            #     result["query_len"][i] + result["pos_len"][i] * (1 + num_hard_negatives)
            # )
            result["total_len"].append(result["query_len"][i] + result["pos_len"][i])

    return result


def msmarco_dataset(
    queries_dataset: Dataset,
    pos_passages_dataset: Dataset,
    tokenizer,
    max_query_len: int = 32,
    max_passage_len: int = 256,
    num_hard_negatives: int = 7,
    sort_by_length: bool = True,
    neg_passages_dataset: Optional[Dataset] = None,
    query_task: str = "Retrieval-query",
    document_task: str = "Retrieval-document",
    batch_size: int = 1000,
    rank=None,
) -> Dataset:
    """
    Prepares MS MARCO dataset with batched processing for efficiency.

    Args:
        queries_dataset: HF Dataset with 'text' column
        pos_passages_dataset: HF Dataset with 'text' column
        tokenizer: HuggingFace tokenizer
        max_query_len: Maximum query length
        max_passage_len: Maximum passage length
        num_hard_negatives: Number of hard negatives per query
        sort_by_length: Whether to sort by total length
        neg_passages_dataset: Optional HF Dataset with 'text' column (list of negatives)
        query_task: Task type for query prompt
        document_task: Task type for document prompt
        batch_size: Batch size for processing

    Returns:
        HF Dataset with tokenized and optionally sorted data
    """

    assert len(queries_dataset) == len(pos_passages_dataset)
    if neg_passages_dataset is not None:
        assert len(queries_dataset) == len(neg_passages_dataset)

    # Get prompts
    query_prompt = TASK_PROMPTS[query_task]
    doc_prompt = TASK_PROMPTS[document_task]
    if rank is None or rank == 0:
        print(f"Query prompt: '{query_prompt}'")
        print(f"Document prompt: '{doc_prompt}'")

    # Combine datasets
    if neg_passages_dataset is not None:
        combined = Dataset.from_dict(
            {
                "query": queries_dataset["query_text"],
                "pos_passage": pos_passages_dataset["positive_text"],
                "pos_ids": pos_passages_dataset["positive_id"],
                "neg_passages": neg_passages_dataset["negative_text"],
                "neg_ids": neg_passages_dataset["nagative_id"],
            }
        )
    else:
        combined = Dataset.from_dict(
            {
                "query": queries_dataset["query_text"],
                "pos_passage": pos_passages_dataset["positive_text"],
                "pos_title": pos_passages_dataset["positive_title"],
                "pos_ids": pos_passages_dataset["positive_id"],
            }
        )

    if rank is None or rank == 0:
        print(f"Tokenizing {len(combined)} examples with batch_size={batch_size}...")

    tokenize_batch = partial(
        tokenize_batch,
        query_prompt=query_prompt,
        doc_prompt=doc_prompt,
        tokenizer=tokenizer,
        max_query_len=max_query_len,
        max_passage_len=max_passage_len,
        num_hard_negatives=num_hard_negatives,
    )

    # Apply batched tokenization
    tokenized_dataset = combined.map(
        tokenize_batch, batched=True, batch_size=batch_size, remove_columns=combined.column_names
    )

    tot_tokens = np.sum(tokenized_dataset["total_len"])
    # Sort by length if requested
    if sort_by_length:
        tokenized_dataset = tokenized_dataset.sort("total_len", reverse=True)

    if rank is None or rank == 0:
        print(f"{tot_tokens/10**6: .1f}M tokens")
        print(f"{len(tokenized_dataset)/10**3: .1f}k query-pas pairs")
        print(f"most long: {tokenized_dataset["total_len"][:30]}")
        print(f"avg query len: {np.mean(tokenized_dataset["query_len"])}")
        print(f"avg doc len: {np.mean(tokenized_dataset["pos_len"])}")

    return tokenized_dataset


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


# def collate_fn_with_padding(batch, pad_token_id=0):
#     """
#     Collate function that pads sequences and creates attention masks.

#     Args:
#         batch: List of examples from dataset
#         pad_token_id: Token ID used for padding (usually 0)

#     Returns:
#         Dict with padded input_ids and attention_masks
#     """
#     query_token_ids = [torch.tensor(item["query_token_ids"]) for item in batch]
#     pos_token_ids = [torch.tensor(item["pos_token_ids"]) for item in batch]
#     pos_ids = torch.cat([torch.tensor([item["pos_ids"]]) for item in batch])

#     # Handle neg_token_ids (list of lists)
#     neg_token_ids = []
#     neg_ids = []
#     for item in batch:
#         neg_token_ids.append([torch.tensor(neg) for neg in item["neg_token_ids"]])
#         neg_ids = torch.cat([torch.tensor(item["pos_ids"]) for item in batch])

#     # Pad queries and create attention masks
#     query_token_ids_padded = pad_sequence(
#         query_token_ids, batch_first=True, padding_value=pad_token_id
#     )
#     query_attention_mask = (query_token_ids_padded != pad_token_id).long()

#     # Pad positive passages and create attention masks
#     pos_token_ids_padded = pad_sequence(pos_token_ids, batch_first=True, padding_value=pad_token_id)
#     pos_attention_mask = (pos_token_ids_padded != pad_token_id).long()

#     # Pad negative passages and create attention masks
#     neg_token_ids_padded = []
#     neg_attention_masks = []

#     for negs in neg_token_ids:
#         # Pad each set of negatives for this example
#         padded_negs = pad_sequence(negs, batch_first=True, padding_value=pad_token_id)
#         attention_mask = (padded_negs != pad_token_id).long()

#         # Ensure all have same number of negatives
#         num_negatives = len(batch[0]["neg_token_ids"])
#         if padded_negs.size(0) < num_negatives:
#             padding_rows = num_negatives - padded_negs.size(0)
#             padding = torch.full(
#                 (padding_rows, padded_negs.size(1)), pad_token_id, dtype=padded_negs.dtype
#             )
#             mask_padding = torch.zeros(
#                 (padding_rows, padded_negs.size(1)), dtype=attention_mask.dtype
#             )
#             padded_negs = torch.cat([padded_negs, padding], dim=0)
#             attention_mask = torch.cat([attention_mask, mask_padding], dim=0)

#         neg_token_ids_padded.append(padded_negs)
#         neg_attention_masks.append(attention_mask)

#     # Stack all negatives: (batch_size, num_negatives, seq_len)
#     neg_token_ids_padded = torch.stack(neg_token_ids_padded)
#     neg_attention_masks = torch.stack(neg_attention_masks)

#     return {
#         "query_token_ids": query_token_ids_padded,
#         "query_attention_mask": query_attention_mask,
#         "pos_token_ids": pos_token_ids_padded,
#         "pos_attention_mask": pos_attention_mask,
#         "pos_ids": pos_ids,
#         "neg_token_ids": neg_token_ids_padded,
#         "neg_attention_mask": neg_attention_masks,
#         "neg_ids": neg_ids,
#     }


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


def load_hard_negatives_datasets(
    base_dir,
    num_hard_negatives,
    tokenizer,
    instruction_template,
    max_query_len=256,
    max_passage_len=512,
    rank=0,
):
    """Load and tokenize all hard-negative parquet datasets under *base_dir*.

    Returns a single concatenated HF Dataset sorted by total sequence length
    (longest first) for length-balanced batching.
    """
    parquet_files = sorted(
        glob.glob(os.path.join(base_dir, "**", "data.parquet"), recursive=True)
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
