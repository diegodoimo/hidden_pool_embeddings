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
from utils.sorted_sampler import LenghtSortedSampler
from pathlib import Path
import time


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

            if self.rank == 0:
                print("tokenizing dataset: num anchors", len(data_split["queries"]))

            queries_dataset = create_dataset(
                dataset=data_split["unique_queries"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.query,
            )

            if self.rank == 0:
                print("tokenizing dataset num docs", len(data_split["corpus"]))

            corpus_dataset = create_dataset(
                dataset=data_split["corpus"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.document,
            )

            if self.rank == 0:
                print(f"number tokenized anchors: {len(queries_dataset)}")
                print(f"number tokenized docs: {len(corpus_dataset)}")

            datasets[task_name] = {
                "dataset": {
                    "queries": data_split["queries"],
                    "positives": data_split["positives"],
                    "unique_queries": queries_dataset,
                    "corpus": corpus_dataset,
                },
            }

        return datasets

    @torch.inference_mode()
    def encode(self, model, loader, prompt_type):

        # distributed sampler will duplicate examples at the end

        indices = None
        if hasattr(loader.sampler, "indices"):
            indices = loader.sampler.indices
            assert isinstance(indices, list)
        

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
            embeddings.append(batch_embeddings.float())

        embeddings = torch.cat(embeddings, dim=0)
        indices = torch.tensor(indices)

        if self.world_size > 1 and prompt_type == PromptType.query:
            gathered = [torch.zeros_like(embeddings) for _ in range(self.world_size)]
            dist.all_gather(gathered, embeddings)
            # Concatenate across ranks for this batch
            embeddings = torch.cat(gathered, dim=0)
            embeddings = embeddings[:num_samples]

            # Also gather indices if we tracked them
            if indices is not None:
                gathered_indices = [torch.zeros_like(indices) for _ in range(self.world_size)]
                dist.all_gather(gathered_indices, indices)
                indices = torch.cat(gathered_indices, dim=0)
                indices = indices[:num_samples]

        # Restore original order
        # Create a mapping from shuffled position to original position
        # Sort back to original order
        if prompt_type == PromptType.query:

            sorted_positions = torch.argsort(indices)
            embeddings = embeddings[sorted_positions]
            return embeddings

        return embeddings, indices

    def search(
        self,
        model,
        query_embeddings,
        corpus_dataset,
        collate_fn,
        top_k=100,
        batch_size=64,
        chunk_size=10**5,
    ):

        N_queries = query_embeddings.shape[0]
        N_corpus = len(corpus_dataset)

        top_scores = torch.full((N_queries, top_k), -float("inf"), device=query_embeddings.device)
        top_indices = torch.full(
            (N_queries, top_k), -1, dtype=torch.long, device=query_embeddings.device
        )

        for chunk_idx in range(0, N_corpus, chunk_size):

            torch.cuda.empty_cache()
            
            subcorpus = corpus_dataset.select(range(chunk_idx, min(chunk_idx + chunk_size, N_corpus)))
            sampler_corpus = LenghtSortedSampler(subcorpus)
            corpus_loader = DataLoader(
                subcorpus,
                sampler=sampler_corpus,
                batch_size=batch_size,
                num_workers=16,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            local_corpus_chunk, local_indicies = self.encode(
                model,
                corpus_loader,
                prompt_type=PromptType.document,
            )
            scores = torch.matmul(query_embeddings, local_corpus_chunk.T)

            chunk_top_scores, chunk_top_indices = torch.topk(
                scores,
                k=min(top_k, scores.shape[1]),  # Use min(top_k, chunk_size)
                dim=1,
                largest=True,
            )

            chunk_absolute_indices = local_indicies[chunk_top_indices] + chunk_idx

            combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
            combined_indices = torch.cat([top_indices, chunk_absolute_indices], dim=1)

            # Find the true global top-k among the combined results
            # Note: We need k=top_k
            top_k_in_combined_scores, top_k_in_combined_indices = torch.topk(
                combined_scores,
                k=top_k,
                dim=1,
                largest=True,
            )

            top_scores = top_k_in_combined_scores
            top_indices = torch.gather(combined_indices, 1, top_k_in_combined_indices)

        # --- 4. Distributed Merging (if using multiple GPUs) ---
        if self.world_size > 1:
            scores_list = [torch.empty_like(top_scores) for _ in range(self.world_size)]
            indices_list = [torch.empty_like(top_indices) for _ in range(self.world_size)]
            
            dist.all_gather(scores_list, top_scores)
            dist.all_gather(indices_list, top_indices)
            
            all_scores = torch.cat(scores_list, dim = 0)
            all_indices = torch.cat(indices_list, dim = 0)

            top_scores, top_indices = torch.topk(
                all_scores,
                k=top_k,
                dim=1,
                largest=True,
            )
            top_indices = torch.gather(all_indices, 1, top_indices)

        return top_scores, top_indices

    def mine_one(
        self,
        dataset,
        model,
        batch_size=8,
        top_k=100,
    ):

        model = model.eval()

        # sampler_queries = None
        # sampler_corpus = None
        # if self.world_size > 1:
        #     sampler_queries = torch.utils.data.distributed.DistributedSampler(
        #         dataset["unique_queries"], shuffle=False, drop_last=False
        #     )
        #     sampler_corpus = torch.utils.data.distributed.DistributedSampler(
        #         dataset["corpus"], shuffle=False, drop_last=False
        #     )

        sampler_queries = LenghtSortedSampler(dataset["unique_queries"])

        print(sampler_queries.indices)
        print(hasattr(sampler_queries, "indices"))

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

        if self.rank == 0:
            start = time.time()
            print(f"building query embeddings")
        query_embeddings = self.encode(model, queries_loader, prompt_type=PromptType.query)

        if self.rank == 0:
            print(f"duration: {time.time()-start}")
            print(f"building document embeddings")
        start = time.time()

        # corpus_embeddings, corpus_indices = self.encode(
        #     model, corpus_loader, prompt_type=PromptType.document
        # )

        # scores = torch.matmul(query_embeddings, corpus_embeddings.T)
        # top_scores, top_indices = torch.topk(
        #     scores,
        #     k=min(top_k + 1, len(scores[1]) if len(scores) > 1 else len(scores[-1])),
        #     dim=1,
        #     largest=True,
        # )

        top_scores, top_indices = self.search(
            model=model,
            query_embeddings=query_embeddings,
            corpus_dataset=dataset["corpus"],
            collate_fn=collate_fn,
            top_k=top_k,
            batch_size=batch_size,
            chunk_size=10**5,
        )

        if self.rank == 0:
            print(f"duration: {time.time()-start}")
            print(f"finding hard negatives")

        start = time.time()

        if self.rank == 0:
            print(f"duration: {time.time()-start}")
            print(f"building negative lists")

        start = time.time()
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        array_ids = np.array(dataset["corpus"]["id"])
        array_texts = np.array(dataset["corpus"]["text"])

        # negative_texts = []
        # negative_indices = []
        # for row_indices in top_indices:
        #     negative_indices.append(list(np.array(dataset["corpus"]["id"])[row_indices[50:65]]))
        #     negative_texts.append(list(np.array(dataset["corpus"]["text"])[row_indices[50:65]]))

        hard_negatives = {}
        unique_query_ids = list(dataset["unique_queries"]["id"])
        for row_indices, q_id in zip(top_indices, unique_query_ids):
            negative_indices = list(array_ids[row_indices[50:65]])
            negative_texts = list(array_texts[row_indices[50:65]])
            hard_negatives[q_id] = {"text": negative_texts, "id": negative_indices}

        if self.rank == 0:
            print(f"duration: {time.time()-start}")
        return hard_negatives

    def mine_negatives(self, model, batch_size=64):

        self.batch_size = batch_size
        for name, task in self.datasets.items():
            new_dataset = {}

            dataset = task["dataset"]
            if self.rank == 0:
                print(f"processing datasets {name}")

            hard_negatives = self.mine_one(
                dataset=dataset,
                model=model,
                batch_size=batch_size,
            )

            negative_texts = [hard_negatives[id_]["text"] for id_ in dataset["queries"]["id"]]
            negative_indices = [hard_negatives[id_]["text"] for id_ in dataset["queries"]["id"]]

            if self.rank == 0:
                print(f"saving dataset")
            # Define schema
            features = Features(
                {
                    "anchor_id": Value("string"),
                    "anchor_text": Value("string"),
                    "positive_id": Value("string"),
                    "positive_text": Value("string"),
                    "negative_id": Sequence(Value("string")),
                    "negative_text": Sequence(Value("string")),
                }
            )

            # Create dataset

            print(len(dataset["queries"]["id"]))
            print(len(dataset["queries"]["text"]))
            print(len(dataset["positives"]["id"]))
            print(len(dataset["positives"]["text"]))
            print(len(negative_texts))
            print(len(negative_indices))
            print(len(negative_indices[0]))

            data = {
                "anchor_id": dataset["queries"]["id"],
                "anchor_text": dataset["queries"]["text"],
                "positive_id": dataset["positives"]["id"],
                "positive_text": dataset["positives"]["text"],
                "negative_id": negative_indices,
                "negative_text": negative_texts,
            }

            dataset = Dataset.from_dict(data, features=features)

            if self.rank == 0:
                dataset.save_to_disk(f"{self.path}/{name}")

            dist.barrier()
