import torch
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from typing import Optional

from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


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

    def tokenize_batch(examples: Dict[str, List]) -> Dict[str, List]:
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
            "query_token_ids": query_encs["input_ids"],
            "pos_token_ids": pos_encs["input_ids"],
            "pos_ids": [int(ids) for ids in examples["pos_ids"]],
            "neg_token_ids": [],
            "neg_ids": [],
            "query_len": [len(ids) for ids in query_encs["input_ids"]],
            "pos_len": [len(ids) for ids in pos_encs["input_ids"]],
            "avg_neg_len": [],
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

    print(f"Tokenizing {len(combined)} examples with batch_size={batch_size}...")
    # Apply batched tokenization
    tokenized_dataset = combined.map(
        tokenize_batch, batched=True, batch_size=batch_size, remove_columns=combined.column_names
    )

    tot_tokens = tokenized_dataset["total_len"].sum()
    print(tot_tokens)

    # Sort by length if requested
    if sort_by_length:
        print("Sorting by total length...")
        tokenized_dataset = tokenized_dataset.sort("total_len")
        print("Sorting complete!")

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
    pos_ids = torch.cat([torch.tensor(item["pos_ids"]) for item in batch])

    # Handle neg_token_ids (list of lists)
    neg_token_ids = []
    neg_ids = []
    for item in batch:
        neg_token_ids.append([torch.tensor(neg) for neg in item["neg_token_ids"]])
        neg_ids = torch.cat([torch.tensor(item["pos_ids"]) for item in batch])

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids, batch_first=True, padding_value=pad_token_id
    )
    query_attention_mask = (query_token_ids_padded != pad_token_id).long()

    # Pad positive passages and create attention masks
    pos_token_ids_padded = pad_sequence(pos_token_ids, batch_first=True, padding_value=pad_token_id)
    pos_attention_mask = (pos_token_ids_padded != pad_token_id).long()

    # Pad negative passages and create attention masks
    neg_token_ids_padded = []
    neg_attention_masks = []

    for negs in neg_token_ids:
        # Pad each set of negatives for this example
        padded_negs = pad_sequence(negs, batch_first=True, padding_value=pad_token_id)
        attention_mask = (padded_negs != pad_token_id).long()

        # Ensure all have same number of negatives
        num_negatives = len(batch[0]["neg_token_ids"])
        if padded_negs.size(0) < num_negatives:
            padding_rows = num_negatives - padded_negs.size(0)
            padding = torch.full(
                (padding_rows, padded_negs.size(1)), pad_token_id, dtype=padded_negs.dtype
            )
            mask_padding = torch.zeros(
                (padding_rows, padded_negs.size(1)), dtype=attention_mask.dtype
            )
            padded_negs = torch.cat([padded_negs, padding], dim=0)
            attention_mask = torch.cat([attention_mask, mask_padding], dim=0)

        neg_token_ids_padded.append(padded_negs)
        neg_attention_masks.append(attention_mask)

    # Stack all negatives: (batch_size, num_negatives, seq_len)
    neg_token_ids_padded = torch.stack(neg_token_ids_padded)
    neg_attention_masks = torch.stack(neg_attention_masks)

    return {
        "query_token_ids": query_token_ids_padded,
        "query_attention_mask": query_attention_mask,
        "pos_token_ids": pos_token_ids_padded,
        "pos_attention_mask": pos_attention_mask,
        "pos_ids": pos_ids,
        "neg_token_ids": neg_token_ids_padded,
        "neg_attention_mask": neg_attention_masks,
        "neg_ids": neg_ids,
    }
