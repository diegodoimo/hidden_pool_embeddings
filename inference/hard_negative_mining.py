import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from inference.create_datasets import create_dataset

from functools import partial
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from datasets import Dataset, Features, Value, Sequence
from .load_datasets import load_data_retrieval
from tasks import get_task
from utils.sorted_sampler import LenghtSortedSampler
from pathlib import Path
import time
from typing import cast
from copy import copy
from mteb.types import HFSubset
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from .helpers import encode, search, collate_fn_with_padding, abs_task_preprocessing, last_token_pool


def estimate_chunk_size(query_embeddings, max_chunk=2*10**5):
    free_mem, _ = torch.cuda.mem_get_info()
    bytes_per_number = query_embeddings.element_size()
    bytes_per_doc = query_embeddings.shape[1] * bytes_per_number
    bytes_per_sim_column = query_embeddings.shape[0] * bytes_per_number
    chunk = int(0.7 * free_mem // (bytes_per_doc + bytes_per_sim_column))
    return max(1000, min(chunk, max_chunk))



class HardNegativesMiner:

    def __init__(self, path, tokenizer, tasks, instruction_template, padding_side="right"):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = tasks
        self.padding_side = padding_side
        if self.rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.instruction_template = instruction_template
        self.path = path
        #self.datasets = self.prepare_datasets(instruction_template)
        dist.barrier()

    # def prepare_datasets(self, instruction_template):

    #     datasets = {}
    #     for task_name in self.task_names:

    #         task = get_task(task_name)
    #         data_split = load_data_retrieval(task)

    #         assert len(data_split["queries"]["text"]) > 1
    #         assert len(data_split["queries"]["text"]) == len(data_split["positives"]["text"])

    #         if self.rank == 0:
    #             print("tokenizing dataset: num anchors", len(data_split["queries"]))

    #         queries_dataset = create_dataset(
    #             dataset=data_split["unique_queries"],
    #             task_metadata=task.metadata,
    #             instruction_template=instruction_template,
    #             tokenizer=self.tokenizer,
    #             prompt_type=PromptType.query,
    #         )

    #         if self.rank == 0:
    #             print("tokenizing dataset num docs", len(data_split["corpus"]))

    #         corpus_dataset = create_dataset(
    #             dataset=data_split["corpus"],
    #             task_metadata=task.metadata,
    #             instruction_template=instruction_template,
    #             tokenizer=self.tokenizer,
    #             prompt_type=PromptType.document,
    #         )

    #         dist.barrier()
    #         if self.rank == 0:
    #             print(f"number tokenized anchors: {len(queries_dataset)}")
    #             print(f"number tokenized docs: {len(corpus_dataset)}")

    #         datasets[task_name] = {
    #             "dataset": {
    #                 "queries": data_split["queries"],
    #                 "positives": data_split["positives"],
    #                 "unique_queries": queries_dataset,
    #                 "corpus": corpus_dataset,
    #             },
    #         }
            

    #     return datasets



    def prepare_dataset(self, task_name):

        task = get_task(task_name)
        data_split = load_data_retrieval(task)

        dist.barrier()
        assert len(data_split["queries"]["text"]) > 1
        assert len(data_split["queries"]["text"]) == len(data_split["positives"]["text"])
        if self.rank == 0:
            print("tokenizing dataset: num anchors", len(data_split["queries"]))

        queries_dataset = create_dataset(
            dataset=data_split["unique_queries"],
            task_metadata=task.metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
        )

        positives_dataset = create_dataset(
            dataset=data_split["unique_positives"],
            task_metadata=task.metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
        )

        dist.barrier()
        if self.rank == 0:
            print("tokenizing dataset num docs", len(data_split["corpus"]))
        corpus_dataset = create_dataset(
            dataset=data_split["corpus"],
            task_metadata=task.metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.document,
        )

        dataset = {
                "queries": data_split["queries"],
                "positives": data_split["positives"],
                "unique_queries": queries_dataset,
                "unique_positives": positives_dataset,
                "corpus": corpus_dataset,
            }
        
        dist.barrier()
        if self.rank == 0:
            print(f"number tokenized anchors: {len(queries_dataset)}")
            print(f"number tokenized docs: {len(corpus_dataset)}")


        return dataset

    def mine_one(
        self,
        dataset,
        model,
        batch_size=8,
        top_k=100,
    ):

        sampler_queries = LenghtSortedSampler(dataset["unique_queries"])
        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
            tokenizer=self.tokenizer,
            eot_id=self.tokenizer.pad_token_id,
        )
        queries_loader = DataLoader(
            dataset["unique_queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        sampler_positives = LenghtSortedSampler(dataset["unique_positives"])
        positives_loader = DataLoader(
            dataset["unique_positives"],
            sampler=sampler_positives,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        dist.barrier()
        if self.rank == 0:
            start = time.time()
            print(f"building query embeddings")

        query_embeddings = encode(
            model,
            queries_loader,
            prompt_type=PromptType.query,
            world_size=self.world_size,
        )

        dist.barrier()
        positive_embeddings = encode(
            model, 
            positives_loader, 
            prompt_type=PromptType.document, 
            world_size=self.world_size,
        )

        query_positive_scores = (query_embeddings*positive_embeddings).sum(dim = 1)
        
        query_positive_scores = query_positive_scores.cpu().numpy()

        del positive_embeddings
        torch.cuda.empty_cache()
        
        dist.barrier()
        chunk_size = estimate_chunk_size(query_embeddings)
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60}min")
            print(f"building document embeddings")
            print(f"selected_chunk_size: {chunk_size}")
        start = time.time()

        top_scores, top_indices = search(
            model=model,
            query_embeddings=query_embeddings,
            corpus_dataset=dataset["corpus"],
            collate_fn=collate_fn,
            top_k=top_k,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60} min")
            print(f"building negative lists")

        start = time.time()
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # array_ids = np.array(dataset["corpus"]["id"])
        # array_texts = np.array(dataset["corpus"]["text"])

        # hard_negatives = {}
        # unique_query_ids = list(dataset["unique_queries"]["id"])
        # for row_indices, row_scores, qp_score, q_id in zip(top_indices, top_scores, unique_query_ids, query_positive_scores):
            
        #     upper_threshold = min(query_positive_score, 0.85)
        #     selected_row_indices = row_indices[5:100]
        #     selected_row_scores = row_scores[5:100]
        #     mask = selected_row_scores < upper_threshold
        #     selected_row_indices = selected_row_indices[mask][:24]

        #     negative_indices = list(array_ids[selected_row_indices])
        #     negative_texts = list(array_texts[selected_row_indices])
        #     hard_negatives[q_id] = {"text": negative_texts, "id": negative_indices}

        hard_negatives = self.get_hard_negatives(
            top_scores=top_scores, 
            top_indices=top_indices, 
            corpus_ids=dataset["corpus"]["id"], 
            corpus_texts=dataset["corpus"]["text"],
            unique_query_ids=dataset["unique_queries"]["id"],
            query_positive_scores=query_positive_scores,
            )

        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60} min")
        return hard_negatives


    def get_hard_negatives(self, top_scores, top_indices, corpus_ids, corpus_texts, unique_query_ids, query_positive_scores):

        array_ids = np.asarray(corpus_ids)
        array_texts = np.asarray(corpus_texts)
        unique_query_ids = np.asarray(unique_query_ids)
        #query_positive_scores = np.asarray(query_positive_scores)

        assert top_scores.shape == top_indices.shape, "Scores / indices shape mismatch"
        assert len(unique_query_ids) == top_scores.shape[0], "Query count mismatch"
        assert len(query_positive_scores) == top_scores.shape[0], "Positive score mismatch"

        candidate_indices = top_indices[:, 5:100]   # (Q, 95)
        candidate_scores = top_scores[:, 5:100]     # (Q, 95)

        # we want the hard negatives to have a similarity lower than 95% the positive and the arbitrary threshold 0.85.
        upper_thresholds = np.minimum(0.95*query_positive_scores, 0.85)[:, None]  # (Q, 1)
        valid_mask = candidate_scores < upper_thresholds   # (Q, 95)
        
        hard_negatives = {}
        discarded = 0
        for i, q_id in enumerate(unique_query_ids):
            valid_idx = np.where(valid_mask[i])[0]

            # Safety: allow empty negatives
            if valid_idx.size < 24:
                discarded += 1 
                if valid_idx.size == 0:
                    hard_negatives[q_id] = {"id": [], "text": []}
                    continue

            # Cap number of negatives
            selected = valid_idx[:24]
            corpus_indices = candidate_indices[i, selected]

            hard_negatives[q_id] = {
                "id": array_ids[corpus_indices].tolist(),
                "text": array_texts[corpus_indices].tolist(),
            }


        if discarded and self.rank ==0:
            print(f"{discarded} examples have less than 24 hard negatives, {discarded/top_scores.shape[0]*100: .2f}%")

        return hard_negatives


    def mine_negatives(self, model, batch_size=64):

        self.batch_size = batch_size
        for name in self.task_names:
            
            if self.rank == 0:
                print(f"preparing dataset {name}\n")

            dataset = self.prepare_dataset(task_name = name)

        #for name, task in self.datasets.items():
            #new_dataset = {}
            #dataset = task["dataset"]

            if self.rank == 0:
                print(f"processing dataset {name}\n")

            hard_negatives = self.mine_one(
                dataset=dataset,
                model=model,
                batch_size=batch_size,
            )
            
            negative_texts = [hard_negatives[id_]["text"] for id_ in dataset["queries"]["id"]]
            negative_indices = [hard_negatives[id_]["id"] for id_ in dataset["queries"]["id"]]
            # for id_ in dataset["queries"]["id"]:
            #     if id_ in hard_negatives:
            #         negative_texts.append(hard_negatives[id_]["text"])
            #         negative_indices.append(hard_negatives[id_]["id"])

            if self.rank == 0:
                print(f"saving dataset {name}\n")

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
