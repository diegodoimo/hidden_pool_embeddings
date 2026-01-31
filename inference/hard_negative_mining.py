import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from inference.create_datasets import create_dataset

from functools import partial
import numpy as np
import torch.distributed as dist

from datasets import Dataset, Features, Value, Sequence
from .load_datasets import load_data_retrieval
from tasks import get_task
from utils.sorted_sampler import LenghtSortedSampler
from pathlib import Path
import time
from .helpers import encode, search, collate_fn_with_padding
from collections import Counter
from dataclasses import dataclass
from datasets import DatasetInfo
import json

def estimate_chunk_size(query_embeddings, max_chunk=5 * 10**4):
    free_mem, _ = torch.cuda.mem_get_info()
    bytes_per_number = query_embeddings.element_size()
    bytes_per_doc = query_embeddings.shape[1] * bytes_per_number
    bytes_per_sim_column = query_embeddings.shape[0] * bytes_per_number
    chunk = int(0.7 * free_mem // (bytes_per_doc + bytes_per_sim_column))
    return max(1000, min(chunk, max_chunk))


@dataclass
class TripletStats:
    less_than_24: int = 0
    less_than_15: int = 0
    less_than_7: int = 0
    empty_entries: int = 0
    total_queries: int = 0

    def update(self, num_negatives: int, q_id_counts):
        """Update stats based on number of negatives"""
        if num_negatives < 24:
            self.less_than_24 += q_id_counts
        if num_negatives < 15:
            self.less_than_15 += q_id_counts
        if num_negatives < 7:
            self.less_than_7 += q_id_counts
        if num_negatives == 0:
            self.empty_entries += q_id_counts
            #if self.rank == 0:
                #print("Found empty entry", q_id_counts)


class HardNegativesMiner:

    def __init__(
        self,
        path,
        model_name,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = tasks
        self.padding_side = padding_side
        if self.rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.instruction_template = instruction_template
        self.path = path
        self.model_name = model_name
        dist.barrier()

    def prepare_dataset(self, task_name):

        task = get_task(task_name)
        data_split, corpus_dict, has_title = load_data_retrieval(task)

        dist.barrier()
        assert len(data_split["queries"]["text"]) > 1
        assert len(data_split["queries"]["text"]) == len(
            data_split["positives"]["text"]
        )
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
            prompt_type=PromptType.document,
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
            print(f"total queries (with repetitions): {len(dataset['queries'])}")
            print(f"number tokenized docs: {len(corpus_dataset)}")

        return dataset, corpus_dict, has_title

    def mine_one(
        self,
        dataset,
        model,
        has_title,
        corpus_dict,
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
            print("building query embeddings")

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

        query_positive_scores = (query_embeddings * positive_embeddings).sum(dim=1)
        query_positive_scores = query_positive_scores.cpu().numpy()

        del positive_embeddings
        torch.cuda.empty_cache()

        dist.barrier()
        chunk_size = estimate_chunk_size(query_embeddings)
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60}min")
            print("building document embeddings")
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
            print("building negative lists")

        start = time.time()
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        q_id_counts = Counter(dataset["queries"]["id"])
        
        
        hard_negatives, stats = self.get_hard_negatives(
            top_scores=top_scores,
            top_indices=top_indices,
            corpus_ids=dataset["corpus"]["id"],
            unique_query_ids=dataset["unique_queries"]["id"],
            query_positive_scores=query_positive_scores,
            has_title=has_title,
            corpus_dict=corpus_dict,
            q_id_counts=q_id_counts,
        )
        stats.total_queries = len(dataset["queries"]["id"])

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60} min")

        return hard_negatives, stats

    def get_hard_negatives(
        self,
        top_scores,
        top_indices,
        corpus_ids,
        unique_query_ids,
        query_positive_scores,
        has_title,
        corpus_dict,
        q_id_counts,
    ):

        array_ids = np.asarray(corpus_ids)
        unique_query_ids = np.asarray(unique_query_ids)
        total_queries = sum(q_id_counts.values())

        if self.rank == 0:
            print("Shapes:", top_scores.shape, top_indices.shape, unique_query_ids.shape, query_positive_scores.shape)
        
        assert (
            top_scores.shape == top_indices.shape
        ), f"Scores / indices shape mismatch {top_scores.shape} {top_indices.shape}"
        assert (
            len(unique_query_ids) == top_scores.shape[0]
        ), f"Query count mismatch {len(unique_query_ids)} {top_scores.shape[0]}"
        assert (
            len(query_positive_scores) == top_scores.shape[0]
        ), f"Positive score mismatch {len(query_positive_scores)} {top_scores.shape[0]}"


        upper_thresholds_relevent_docs = min(150, int(0.1* len(corpus_ids)))

        candidate_indices = top_indices[:, 5:upper_thresholds_relevent_docs]  # (Q, 95)
        candidate_scores = top_scores[:, 5:upper_thresholds_relevent_docs]  # (Q, 95)
        # we want the hard negatives to have a similarity lower than 95% the positive and the arbitrary threshold.
        upper_thresholds = np.minimum(0.95 * query_positive_scores, 0.9)[
            :, None
        ]  # (Q, 1)

        if self.rank == 0:
            print("upper_thresholds", upper_thresholds[:10])
            print("candidate_scores", candidate_scores[:10, :10])
        valid_mask = candidate_scores < upper_thresholds  # (Q, 95)

        stats = TripletStats()
        hard_negatives = {}

        for i, q_id in enumerate(unique_query_ids):

            valid_idx = np.where(valid_mask[i])[0]
            if self.rank == 0:
                #print("q_id:", q_id)
                #print("candidate scores:", candidate_scores[i])
                #print("valid mask:", valid_mask[i])
                #print("valid idx:", valid_idx)
                print("num valid negatives:", valid_idx.size)

            stats.update(valid_idx.size, q_id_counts[q_id])

            if valid_idx.size == 0:
                # Use the same structure as the normal case, just with empty lists
                if has_title:
                    hard_negatives[q_id] = {"id": [], "text": [], "title": []}
                else:
                    hard_negatives[q_id] = {"id": [], "text": []}
                continue

            # Cap number of negatives
            selected = valid_idx[:24]
            corpus_indices = candidate_indices[i, selected]

            corpus_ids = array_ids[corpus_indices].tolist()

            if has_title:
                hard_negatives[q_id] = {
                    "id": corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in corpus_ids],
                    "title": [corpus_dict[id_]["title"] for id_ in corpus_ids],
                }

            else:
                hard_negatives[q_id] = {
                    "id": corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in corpus_ids],
                }

            # DRAMATIC DROP IN PERFORMANCE FOR THE INDEXING OF CORPUS TEXT LIST
            # hard_negatives[q_id] = {
            #     "id": corpus_ids,
            #     "text": [corpus_texts[index] for index in corpus_indices],
            # }

        if stats.less_than_24 and self.rank == 0:
            print("total elements:", total_queries)
            print(
                f"{stats.less_than_24} examples have less than 24 hard negatives, {stats.less_than_24/total_queries*100: .2f}%, \
                    {stats.empty_entries/total_queries*100: .2f}% are empty"
            )

        return hard_negatives, stats

    def mine_negatives(self, model, batch_size=64):

        self.batch_size = batch_size
        for task_name in self.task_names:

            if self.rank == 0:
                print(f"preparing dataset {task_name}\n")

            dataset, corpus_dict, has_title = self.prepare_dataset(task_name=task_name)

            if self.rank == 0:
                print(f"processing dataset {task_name}\n")

            hard_negatives, stats = self.mine_one(
                dataset=dataset,
                model=model,
                batch_size=batch_size,
                has_title=has_title,
                corpus_dict=corpus_dict,
            )

            dist.barrier()
            if self.rank == 0:
                print(f"saving dataset {task_name}\n")

            self.save_to_disk(
                dataset=dataset,
                negatives=hard_negatives,
                has_title=has_title,
                stats=stats,
                task_name=task_name,                
            )
            

    def save_to_disk(self, dataset, negatives, has_title, stats, task_name):

        texts = dataset["queries"]["text"]
        ids = dataset["queries"]["id"]
        positive_text = dataset["positives"]["text"]
        positive_id = dataset["positives"]["id"]
        negative_text = [negatives[id_]["text"] for id_ in dataset["queries"]["id"]]
        negative_id = [negatives[id_]["id"] for id_ in dataset["queries"]["id"]]

        if has_title:
            negative_title = [
                negatives[id_]["title"] for id_ in dataset["queries"]["id"]
            ]
            positive_title = dataset["positives"]["title"]

            dataset = Dataset.from_dict(
                {
                    "anchor_text": texts,
                    "anchor_id": ids,
                    "positive_text": positive_text,
                    "positive_title": positive_title,
                    "positive_id": positive_id,
                    "negative_text": negative_text,
                    "negative_title": negative_title,
                    "negative_id": negative_id,
                },
                features=Features(
                    {
                        "anchor_text": Value("string"),
                        "anchor_id": Value("string"),
                        "positive_text": Value("string"),
                        "positive_title": Value("string"),
                        "positive_id": Value("string"),
                        "negative_text": Sequence(Value("string")),
                        "negative_title": Sequence(Value("string")),
                        "negative_id": Sequence(Value("string")),
                    }
                ),
            )
        else:
            dataset = Dataset.from_dict(
                {
                    "anchor_text": texts,
                    "anchor_id": ids,
                    "positive_text": positive_text,
                    "positive_id": positive_id,
                    "negative_text": negative_text,
                    "negative_id": negative_id,
                },
                features=Features(
                    {
                        "anchor_text": Value("string"),
                        "anchor_id": Value("string"),
                        "positive_text": Value("string"),
                        "positive_id": Value("string"),
                        "negative_text": Sequence(Value("string")),
                        "negative_id": Sequence(Value("string")),
                    }
                ),
            )

        # Save metadata
        metadata = {
            "num_triples": stats.total_queries,
            "num_empty_negative_entries": stats.empty_entries,
            "num_with_7_hard_negatives": stats.total_queries - stats.less_than_7,
            "num_with_15_hard_negatives": stats.total_queries - stats.less_than_15,
            "num_with_24_hard_negatives": stats.total_queries - stats.less_than_24,
            "embedder": self.model_name,
        }

        dist.barrier()
        if self.rank == 0:
            dataset.save_to_disk(f"{self.path}/{task_name}")

            with open(f"{self.path}/{task_name}/dataset_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

