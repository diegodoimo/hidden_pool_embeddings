import torch
from torch.utils.data import DataLoader
from mteb.types import PromptType
from inference.create_datasets import create_dataset, filter_paired_datasets_by_length

from functools import partial
import numpy as np
import torch.distributed as dist

from datasets import Dataset, Features, Value, Sequence
from tasks.load_datasets import load_task_data
from tasks import get_task
from utils.sorted_sampler import LenghtSortedSampler
from pathlib import Path
import time
from .helpers import encode, search, collate_fn_with_padding
from collections import Counter
from dataclasses import dataclass
from datasets import DatasetInfo
import json
from utils.helpers import print_memory_consumed


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
            # if self.rank == 0:
            # print("Found empty entry", q_id_counts)


class HardNegativesMiner:

    def __init__(
        self,
        path,
        model_name,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
        max_length=512,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.tokenizer = tokenizer
        self.task_names = tasks
        self.padding_side = padding_side
        self.max_length = max_length

        if self.rank == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.instruction_template = instruction_template
        self.path = path
        self.model_name = model_name
        dist.barrier()

    def prepare_dataset(self, task_name):

        task = get_task(task_name)
        data_split, corpus_dict, has_title = load_task_data(task)

        dist.barrier()
        assert len(data_split["queries"]["text"]) > 1
        assert len(data_split["queries"]["text"]) == len(
            data_split["positives"]["text"]
        )
        if self.rank == 0:
            print("tokenizing dataset: num anchors", len(data_split["queries"]))

        # Filter queries and positives by length while maintaining pair correspondence
        # (
        #     unique_queries_filtered,
        #     unique_positives_filtered,
        #     filtered_queries,
        #     filtered_positives,
        # ) = filter_paired_datasets_by_length(
        #     unique_queries_dataset=data_split["unique_queries"],
        #     unique_positives_dataset=data_split["unique_positives"],
        #     queries_with_reps=data_split["queries"],
        #     positives_with_reps=data_split["positives"],
        #     tokenizer=self.tokenizer,
        #     instruction_template=self.instruction_template,
        #     task_metadata=task.metadata,
        #     max_length=self.max_length,
        #     rank=self.rank,
        # )

        unique_queries_dataset = create_dataset(
            dataset=data_split["unique_queries"],
            task_metadata=task.metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
            max_length=self.max_length,
        )

        unique_positives_dataset = create_dataset(
            dataset=data_split["unique_positives"],
            task_metadata=task.metadata,
            instruction_template=self.instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.document,
            max_length=self.max_length,
        )

        filtered_positives, filtered_queries = filter_paired_datasets_by_length(
            unique_queries_dataset.removed_ids,
            unique_positives_dataset.removed_ids,
            data_split["queries"],
            data_split["positives"],
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
            max_length=self.max_length,
        )

        dataset = {
            "queries": filtered_queries,
            "positives": filtered_positives,
            "unique_queries": unique_queries_dataset,
            "unique_positives": unique_positives_dataset,
            "corpus": corpus_dataset,
        }

        dist.barrier()
        if self.rank == 0:
            print(f"number unique queries: {len(unique_queries_dataset)}")
            print(f"number unique positives: {len(unique_positives_dataset)}")
            print(f"number of documents: {len(corpus_dataset)}")

            print(f"total queries (with repetitions): {len(dataset['queries']['id'])}")
            print(
                f"total positives (with repetitions): {len(dataset['positives']['id'])}"
            )

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
        if self.rank == 0:
            print(f"queries embedding duration: {(time.time()-start)/60} min")
            start = time.time()
            print("building query positive embeddings")

        positive_embeddings = encode(
            model,
            positives_loader,
            prompt_type=PromptType.document,
            world_size=self.world_size,
        )

        if self.rank == 0:
            print(f"positive embedding duration: {(time.time()-start)/60} min")

        # Create mappings from IDs to embedding indices
        unique_query_id_to_idx = {
            qid: idx for idx, qid in enumerate(dataset["unique_queries"]["id"])
        }
        unique_positive_id_to_idx = {
            pid: idx for idx, pid in enumerate(dataset["unique_positives"]["id"])
        }

        # Map each (query, positive) pair in qrels to their corresponding embeddings
        query_indices = [
            unique_query_id_to_idx[qid] for qid in dataset["queries"]["id"]
        ]
        positive_indices = [
            unique_positive_id_to_idx[pid] for pid in dataset["positives"]["id"]
        ]

        # Expand embeddings to match qrels order
        expanded_query_embeddings = query_embeddings[query_indices]
        expanded_positive_embeddings = positive_embeddings[positive_indices]

        # Compute scores for each (query, positive) pair in qrels
        query_positive_scores = (
            expanded_query_embeddings * expanded_positive_embeddings
        ).sum(dim=1)
        query_positive_scores = query_positive_scores.cpu().numpy()

        del positive_embeddings
        torch.cuda.empty_cache()

        dist.barrier()
        chunk_size = estimate_chunk_size(query_embeddings)
        if self.rank == 0:
            print(f"query positive scopre duration: {(time.time()-start)/60}min")
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

        del query_embeddings
        torch.cuda.empty_cache()

        dist.barrier()
        if self.rank == 0:
            print(f"duration: {(time.time()-start)/60} min")
            print("building negative lists")

        start = time.time()
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        hard_negatives, stats = self.get_hard_negatives(
            top_scores=top_scores,
            top_indices=top_indices,
            corpus_ids=dataset["corpus"]["id"],
            query_ids=dataset["queries"]["id"],
            positive_ids=dataset["positives"]["id"],
            unique_query_ids=dataset["unique_queries"]["id"],
            query_positive_scores=query_positive_scores,
            unique_query_id_to_idx=unique_query_id_to_idx,
            has_title=has_title,
            corpus_dict=corpus_dict,
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
        query_ids,
        positive_ids,
        unique_query_ids,
        query_positive_scores,
        unique_query_id_to_idx,
        has_title,
        corpus_dict,
    ):

        array_ids = np.asarray(corpus_ids)
        unique_query_ids = np.asarray(unique_query_ids)
        total_queries = len(query_ids)

        if self.rank == 0:
            print(
                "Shapes:",
                top_scores.shape,
                top_indices.shape,
                unique_query_ids.shape,
                query_positive_scores.shape,
            )

        assert (
            top_scores.shape == top_indices.shape
        ), f"Scores / indices shape mismatch {top_scores.shape} {top_indices.shape}"
        assert (
            len(unique_query_ids) == top_scores.shape[0]
        ), f"Query count mismatch {len(unique_query_ids)} {top_scores.shape[0]}"
        assert (
            len(query_positive_scores) == total_queries
        ), f"Positive score mismatch {len(query_positive_scores)} {total_queries}"

        upper_thresholds_relevent_docs = min(150, int(0.1 * len(corpus_ids)))

        stats = TripletStats()
        hard_negatives = {}

        # Process each (query, positive) pair in qrels
        for qrel_idx, (q_id, p_id) in enumerate(zip(query_ids, positive_ids)):
            # Get the unique query index for this qrel entry
            unique_q_idx = unique_query_id_to_idx[q_id]

            # Get candidate documents for this query
            candidate_indices = top_indices[
                unique_q_idx, 5:upper_thresholds_relevent_docs
            ]
            candidate_scores = top_scores[
                unique_q_idx, 5:upper_thresholds_relevent_docs
            ]

            # Threshold based on this specific (query, positive) pair's score
            upper_threshold = min(0.95 * query_positive_scores[qrel_idx], 0.9)

            # Find valid hard negatives
            valid_mask = candidate_scores < upper_threshold
            valid_idx = np.where(valid_mask)[0]

            # if self.rank == 0 and qrel_idx < 10:
            #     print(
            #         f"qrel {qrel_idx}: q_id={q_id}, p_id={p_id}, "
            #         f"score={query_positive_scores[qrel_idx]:.4f}, "
            #         f"threshold={upper_threshold:.4f}, "
            #         f"num_valid={valid_idx.size}"
            #     )

            stats.update(valid_idx.size, 1)

            # Use composite key (query_id, positive_id) to store negatives per qrels entry
            key = (q_id, p_id)

            if valid_idx.size == 0:
                # Use the same structure as the normal case, just with empty lists
                if has_title:
                    hard_negatives[key] = {"id": [], "text": [], "title": []}
                else:
                    hard_negatives[key] = {"id": [], "text": []}
                continue

            # Cap number of negatives
            selected = valid_idx[:24]
            corpus_indices = candidate_indices[selected]

            neg_corpus_ids = array_ids[corpus_indices].tolist()

            if has_title:
                hard_negatives[key] = {
                    "id": neg_corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in neg_corpus_ids],
                    "title": [corpus_dict[id_]["title"] for id_ in neg_corpus_ids],
                }

            else:
                hard_negatives[key] = {
                    "id": neg_corpus_ids,
                    "text": [corpus_dict[id_]["text"] for id_ in neg_corpus_ids],
                }

            # DRAMATIC DROP IN PERFORMANCE FOR THE INDEXING OF CORPUS TEXT LIST
            # hard_negatives[q_id] = {
            #     "id": corpus_ids,
            #     "text": [corpus_texts[index] for index in corpus_indices],
            # }

        if self.rank == 0:
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

            dist.barrier()
            if self.rank == 0:
                print(f"processing dataset {task_name}\n")
                print_memory_consumed(rank=self.rank)

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

            torch.cuda.empty_cache()
            print_memory_consumed(rank=self.rank)

    def save_to_disk(self, dataset, negatives, has_title, stats, task_name):

        texts = dataset["queries"]["text"]
        ids = dataset["queries"]["id"]
        positive_text = dataset["positives"]["text"]
        positive_id = dataset["positives"]["id"]

        # Use (query_id, positive_id) tuples to get negatives for each qrels entry
        negative_text = [
            negatives[(q_id, p_id)]["text"]
            for q_id, p_id in zip(dataset["queries"]["id"], dataset["positives"]["id"])
        ]
        negative_id = [
            negatives[(q_id, p_id)]["id"]
            for q_id, p_id in zip(dataset["queries"]["id"], dataset["positives"]["id"])
        ]

        if has_title:
            negative_title = [
                negatives[(q_id, p_id)]["title"]
                for q_id, p_id in zip(
                    dataset["queries"]["id"], dataset["positives"]["id"]
                )
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
