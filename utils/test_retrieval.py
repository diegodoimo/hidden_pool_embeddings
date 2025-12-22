import mteb
import torch
from torch.utils.data import DataLoader
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType
from ._create_dataloaders import create_dataset

from typing import cast
from copy import copy
from mteb.types import HFSubset
from datasets import DatasetDict
from functools import partial
import numpy as np
import torch.distributed as dist
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from mteb._evaluators.retrieval_metrics import make_score_dict


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


def collate_fn_with_padding(batch, pad_token_id=0):

    query_token_ids = [torch.tensor(item["input_ids"]) for item in batch]

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids, batch_first=True, padding_value=pad_token_id
    )
    query_attention_mask = (query_token_ids_padded != pad_token_id).long()
    assert query_token_ids_padded.dtype == torch.int64, batch
    return {
        "input_ids": query_token_ids_padded,
        "attention_mask": query_attention_mask,
    }


class evaluate_retrieval:

    def __init__(self, tokenizer, tasks, instruction_template):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tokenizer = tokenizer
        self.task_names = tasks
        self.datasets = self.prepare_datasets(instruction_template)

    def prepare_datasets(self, instruction_template, max_passage_len=4096):

        datasets = {}
        for task_name in self.task_names:
            task = mteb.get_task(task_name)

            eval_splits = task.metadata.eval_splits
            self.main_score = task.metadata.main_score
            assert len(eval_splits) == 1
            eval_split = eval_splits[0]

            self.hf_split = eval_split
            task.load_data()
            task.convert_v1_dataset_format_to_v2()

            data_split, self.hf_subset = abs_task_preprocessing(task, eval_split)

            data_split["relevant_docs"], data_split["queries"] = _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )

            queries_dataset = create_dataset(
                self.tokenizer,
                dataset=data_split["queries"],
                task_metadata=task.metadata,
                prompt_type=PromptType.query,
                instruction_template=instruction_template,
            )
            print(queries_dataset[0]["text"])
            corpus_dataset = create_dataset(
                self.tokenizer,
                dataset=data_split["corpus"],
                task_metadata=task.metadata,
                prompt_type=PromptType.document,
                instruction_template=instruction_template,
            )
            print(corpus_dataset[0]["text"])

            datasets[task_name] = {
                "dataset": {
                    "queries": queries_dataset,
                    "corpus": corpus_dataset,
                    "relevant_docs": data_split["relevant_docs"],
                },
                "task_specific_scores": task.task_specific_scores,
            }

        return datasets

    @torch.inference_mode()
    def encode(self, model, loader):

        # distributed sampler will duplicate examples at the end
        num_samples = len(loader.dataset)
        embeddings = []

        for batch in loader:
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
                out_embeddings = F.normalize(out_embeddings, p=2, dim=1)

            gathered = [torch.zeros_like(out_embeddings) for _ in range(self.world_size)]
            dist.all_gather(gathered, out_embeddings)

            # Concatenate across ranks for this batch
            batch_embeddings = torch.cat(gathered, dim=0)
            embeddings.append(batch_embeddings.float())

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings[:num_samples]

    def evaluate_one(
        self,
        dataset,
        model,
        task_specific_scores,
        batch_size=8,
        top_k=None,
        k_values=[1, 3, 5, 10, 20, 100, 1000],
        ignore_identical_ids: bool = False,
        skip_first_result: bool = False,
    ):

        if top_k is None:
            top_k = max(k_values)

        model = model.eval()

        query_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["queries"]["id"])}
        doc_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["corpus"]["id"])}

        sampler_queries = None
        sampler_corpus = None
        if self.world_size > 1:
            sampler_queries = torch.utils.data.distributed.DistributedSampler(
                dataset["queries"], shuffle=False, drop_last=False
            )
            sampler_corpus = torch.utils.data.distributed.DistributedSampler(
                dataset["corpus"], shuffle=False, drop_last=False
            )

        collate_fn = partial(collate_fn_with_padding, pad_token_id=self.tokenizer.pad_token_id)

        queries_loader = DataLoader(
            dataset["queries"],
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
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

        del scores
        top_scores = top_scores.cpu()
        top_indices = top_indices.tolist()

        results = {}
        for i in range(len(top_scores)):

            results[query_idx_to_id[i]] = {
                doc_idx_to_id[index]: top_scores[i, j].item()
                for j, index in enumerate(top_indices[i])
            }

        qrels = dataset["relevant_docs"]
        if ignore_identical_ids:
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
        ) = calculate_retrieval_scores(results, qrels, list(k_values), skip_first_result)

        task_specific_scores_ = task_specific_scores(
            all_scores,
            dataset["relevant_docs"],
            results,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
        )
        _previous_results_model_meta = None
        scores = make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores_,
            _previous_results_model_meta,
        )
        scores["main_score"] = scores[self.main_score]
        return {self.main_score: scores[self.main_score]}

    def evaluate(self, model, batch_size=64):
        results = {}
        for name, task in self.datasets.items():
            if self.rank == 0:
                print(f"processing datasets {name}")

            results[name] = self.evaluate_one(
                dataset=task["dataset"],
                model=model,
                task_specific_scores=task["task_specific_scores"],
                batch_size=batch_size,
            )

        return results
