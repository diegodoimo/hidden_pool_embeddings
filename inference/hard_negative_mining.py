import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from inference.create_datasets import create_dataset

from typing import cast
from copy import copy
from mteb.types import HFSubset
from datasets import DatasetDict
from functools import partial
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset, Features, Value, Sequence
from .load_datasets import load_data_retrieval
from tasks import get_task
from pathlib import Path


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def abs_task_preprocessing(task, eval_split):

    subsets_to_run = None
    task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)

    if task.hf_subsets is None:
        hf_subsets = list(task.dataset.keys())
    else:
        hf_subsets = copy(task.hf_subsets)

    if subsets_to_run is not None:  # allow overwrites of pre-filtering
        hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

    for hf_subset in hf_subsets:
        if hf_subset not in task.dataset and hf_subset == "default":
            data_split = task.dataset[eval_split]
        else:
            data_split = task.dataset[hf_subset][eval_split]
    assert len(hf_subsets) == 1, hf_subsets
    return data_split, hf_subset


def collate_fn_with_padding(batch, pad_token_id=0, padding_side="right"):

    query_token_ids = [torch.tensor(item["input_ids"]) for item in batch]
    query_attention_mask = [torch.ones_like(input_ids) for input_ids in query_token_ids]

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )

    query_attention_mask = pad_sequence(
        query_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )
    # query_attention_mask = (query_token_ids_padded != pad_token_id).long()
    # query_attention_mask[:, -1]=1

    assert query_token_ids_padded.dtype == torch.int64, batch
    return {
        "input_ids": query_token_ids_padded,
        "attention_mask": query_attention_mask,
    }


class HardNegativesMiner:

    def __init__(self, path, tokenizer, tasks, instruction_template, padding_side="right"):

        self.world_size = dist.get_world_size()
        assert self.world_size == 1
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = tasks
        self.padding_side = padding_side
        if self.rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.path = path
        self.datasets = self.prepare_datasets(instruction_template)
        dist.barrier()

    def prepare_datasets(self, instruction_template):

        datasets = {}
        for task_name in self.task_names:

            task = get_task(task_name)

            data_split = load_data_retrieval(task)

            assert len(data_split["queries"]["text"]) > 1
            assert len(data_split["queries"]["text"]) == len(data_split["positives"]["text"])

            # task.convert_v1_dataset_format_to_v2()
            # data_split, hf_subset = abs_task_preprocessing(task, eval_split)

            # data_split["relevant_docs"], data_split["queries"] = _filter_queries_without_positives(
            #    data_split["relevant_docs"], data_split["queries"]
            # )

            queries_dataset = create_dataset(
                dataset=data_split["queries"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.query,
            )

            corpus_dataset = create_dataset(
                dataset=data_split["corpus"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.document,
            )

            datasets[task_name] = {
                "dataset": {
                    "queries": queries_dataset,
                    "corpus": corpus_dataset,
                },
                "ignore_identical_ids": task.ignore_identical_ids,
            }

        return datasets

    @torch.inference_mode()
    def encode(self, model, loader):

        # distributed sampler will duplicate examples at the end
        num_samples = len(loader.dataset)
        embeddings = []

        for i, batch in enumerate(loader):
            batch = {key: val.to(model.device) for key, val in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_embeddings = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                out_embeddings = last_token_pool(
                    out_embeddings.last_hidden_state,
                    batch["attention_mask"],
                )

                batch_embeddings = F.normalize(out_embeddings, p=2, dim=1)

            # gathered = [torch.zeros_like(out_embeddings) for _ in range(self.world_size)]
            # dist.all_gather(gathered, out_embeddings)

            # Concatenate across ranks for this batch
            # batch_embeddings = torch.cat(gathered, dim=0)
            embeddings.append(batch_embeddings.float())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings[:num_samples]

    def mine_one(
        self,
        dataset,
        model,
        batch_size=8,
        top_k=None,
    ):

        top_k = 100
        model = model.eval()

        sampler_queries = None
        sampler_corpus = None
        if self.world_size > 1:
            sampler_queries = torch.utils.data.distributed.DistributedSampler(
                dataset["unique_queries"], shuffle=False, drop_last=False
            )
            sampler_corpus = torch.utils.data.distributed.DistributedSampler(
                dataset["corpus"], shuffle=False, drop_last=False
            )

        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
        )

        queries_loader = DataLoader(
            dataset["unique_queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        corpus_loader = DataLoader(
            dataset["corpus"],
            sampler=sampler_corpus,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        query_embeddings = self.encode(model, queries_loader)
        corpus_embeddings = self.encode(model, corpus_loader)

        scores = torch.matmul(query_embeddings, corpus_embeddings.T)
        top_scores, top_indices = torch.topk(
            scores,
            k=min(top_k + 1, len(scores[1]) if len(scores) > 1 else len(scores[-1])),
            dim=1,
            largest=True,
        )

        # negative_texts = []
        # negative_indices = []
        # for row_indices in top_indices:
        #     negative_indices.append(list(np.array(dataset["corpus"]["id"])[row_indices[50:65]]))
        #     negative_texts.append(list(np.array(dataset["corpus"]["text"])[row_indices[50:65]]))

        hard_negatives = {}
        unique_query_ids = list(dataset["unique_queries"]["id"])
        for row_indices, q_id in zip(top_indices, unique_query_ids):
            negative_indices = list(np.array(dataset["corpus"]["id"])[row_indices[50:65]])
            negative_texts = list(np.array(dataset["corpus"]["text"])[row_indices[50:65]])
            hard_negatives[q_id] = {"text": negative_texts, "id": negative_indices}

        return hard_negatives

    def mine_negatives(self, model, batch_size=64):

        for name, task in self.datasets.items():
            new_dataset = {}

            if self.rank == 0:
                print(f"processing datasets {name}")

            negative_texts, negative_indices = self.mine_one(
                dataset=task["dataset"],
                ignore_identical_ids=task["ignore_identical_ids"],
                hf_split=task["hf_split"],
                hf_subset=task["hf_subset"],
                main_score=task["main_score"],
                model=model,
                batch_size=batch_size,
            )

            # Define schema
            features = Features(
                {
                    "anchor_id": Value("string"),
                    "anchor_text": Value("string"),
                    "positive_id": Value("string"),
                    "positive_text": Value("string"),
                    "negative_ids": Sequence(Value("string")),
                    "negative_texts": Sequence(Value("string")),
                }
            )

            # Create dataset

            print(len(task["dataset"]["anchor_id"]))
            print(len(task["dataset"]["anchor_text"]))
            print(len(task["dataset"]["positive_id"]))
            print(len(task["dataset"]["positive_text"]))
            print(len(negative_texts))
            print(len(negative_indices))
            print(len(negative_indices[0]))

            data = {
                "anchor_id": task["dataset"]["anchor_id"],
                "anchor_text": task["dataset"]["anchor_text"],
                "positive_id": task["dataset"]["positive_id"],
                "positive_text": task["dataset"]["positive_text"],
                "negative_ids": negative_texts,  # List of lists
                "negative_texts": negative_indices,
            }

            dataset = Dataset.from_dict(data, features=features)

            if self.rank == 0:
                dataset.save_to_disk(f"{self.path}/{name}")

            dist.barrier()
