import mteb
import torch
from torch.utils.data import DataLoader
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType
from .create_datasets import create_dataset

from functools import partial
import torch.distributed as dist
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
import torch.nn.functional as F
from mteb._evaluators.retrieval_metrics import make_score_dict
from utils.sorted_sampler import LenghtSortedSampler
from .helpers import (
    search,
    encode,
    collate_fn_with_padding,
    abs_task_preprocessing,
    last_token_pool,
)
import time


class evaluate_retrieval:

    def __init__(
        self,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
        new_inference_mode=True,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tokenizer = tokenizer
        self.task_names = tasks
        self.datasets = self.prepare_datasets(instruction_template)
        self.padding_side = padding_side
        self.new_inference_mode = new_inference_mode

    def prepare_datasets(self, instruction_template, max_passage_len=4096):

        datasets = {}
        for task_name in self.task_names:
            task = mteb.get_task(task_name)

            eval_splits = task.metadata.eval_splits
            assert len(eval_splits) == 1
            eval_split = eval_splits[0]
            task.load_data()
            task.convert_v1_dataset_format_to_v2()

            data_split, hf_subset = abs_task_preprocessing(task, eval_split)

            data_split["relevant_docs"], data_split["queries"] = (
                _filter_queries_without_positives(
                    data_split["relevant_docs"], data_split["queries"]
                )
            )

            queries_dataset = create_dataset(
                dataset=data_split["queries"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.query,
                max_length=8192,
            )

            corpus_dataset = create_dataset(
                dataset=data_split["corpus"],
                task_metadata=task.metadata,
                instruction_template=instruction_template,
                tokenizer=self.tokenizer,
                prompt_type=PromptType.document,
                max_length=8192,
            )

            datasets[task_name] = {
                "dataset": {
                    "queries": queries_dataset,
                    "corpus": corpus_dataset,
                    "relevant_docs": data_split["relevant_docs"],
                },
                "task_specific_scores": task.task_specific_scores,
                "ignore_identical_ids": task.ignore_identical_ids,
                "hf_split": eval_split,
                "main_score": task.metadata.main_score,
                "hf_subset": hf_subset,
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

            if self.world_size > 1:
                gathered = [
                    torch.zeros_like(out_embeddings) for _ in range(self.world_size)
                ]
                dist.all_gather(gathered, out_embeddings)

                # Concatenate across ranks for this batch
                batch_embeddings = torch.cat(gathered, dim=0)
            embeddings.append(batch_embeddings.float())

        embeddings = torch.cat(embeddings, dim=0)
        return embeddings[:num_samples]

    def evaluate_one(
        self,
        dataset,
        task_specific_scores,
        ignore_identical_ids,
        hf_split,
        hf_subset,
        main_score,
        model,
        batch_size=8,
        top_k=None,
        k_values=[1, 3, 5, 10, 20, 50],
        skip_first_result: bool = False,
    ):

        if top_k is None:
            top_k = max(k_values)
        model = model.eval()

        query_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["queries"]["id"])}
        doc_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["corpus"]["id"])}

        print(f"rank {self.rank}: query_idx_to_id: {len(query_idx_to_id)}")
        print(f"rank {self.rank}: doc_idx_to_id: {len(query_idx_to_id)}")

        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
        )

        sampler_queries = None
        sampler_corpus = None
        if self.world_size > 1:
            sampler_queries = torch.utils.data.distributed.DistributedSampler(
                dataset["queries"], shuffle=False, drop_last=False
            )
            sampler_corpus = torch.utils.data.distributed.DistributedSampler(
                dataset["corpus"], shuffle=False, drop_last=False
            )

        if self.new_inference_mode:
            sampler_queries = LenghtSortedSampler(dataset["queries"])

        if self.rank == 0:
            print("num queries", len(dataset["queries"]))
            print("num documents", len(dataset["queries"]))

        queries_loader = DataLoader(
            dataset["queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if self.new_inference_mode:
            dist.barrier()
            start = time.time()
            query_embeddings = encode(
                model,
                queries_loader,
                prompt_type=PromptType.query,
                world_size=self.world_size,
            )
            dist.barrier()
            if self.rank == 0:
                print(f"time to encode queries: {time.time()-start}")
            time.time()

            top_scores, top_indices = search(
                model=model,
                query_embeddings=query_embeddings,
                corpus_dataset=dataset["corpus"],
                collate_fn=collate_fn,
                top_k=top_k,
                batch_size=batch_size,
                chunk_size=2 * 10**3,
            )

            dist.barrier()
            if self.rank == 0:
                print(f"time to encode docs + search: {time.time()-start}")
        else:
            corpus_loader = DataLoader(
                dataset["corpus"],
                sampler=sampler_corpus,
                batch_size=batch_size,
                num_workers=16,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            dist.barrier()
            start = time.time()
            query_embeddings = self.encode(model, queries_loader)
            dist.barrier()
            if self.rank == 0:
                print(f"time to encode queries: {time.time()-start}")
            time.time()
            corpus_embeddings = self.encode(model, corpus_loader)

            scores = torch.matmul(query_embeddings, corpus_embeddings.T)
            top_scores, top_indices = torch.topk(
                scores,
                k=min(
                    top_k + 1, len(scores[1]) if len(scores) > 1 else len(scores[-1])
                ),
                dim=1,
                largest=True,
            )
            dist.barrier()
            if self.rank == 0:
                print(f"time to encode docs + search: {time.time()-start}")

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
            # Remove identical ids from results dict in some datasets the queries are also in the documents so they must be removed.
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
        ) = calculate_retrieval_scores(
            results,
            qrels,
            list(k_values),
            skip_first_result,
        )

        task_specific_scores_ = task_specific_scores(
            all_scores,
            dataset["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
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
        return {main_score: scores[main_score]}

    def evaluate(self, model, batch_size=64):
        results = {}
        for name, task in self.datasets.items():
            if self.rank == 0:
                print(f"processing datasets {name}")

            results[name] = self.evaluate_one(
                dataset=task["dataset"],
                task_specific_scores=task["task_specific_scores"],
                ignore_identical_ids=task["ignore_identical_ids"],
                hf_split=task["hf_split"],
                hf_subset=task["hf_subset"],
                main_score=task["main_score"],
                model=model,
                batch_size=batch_size,
            )

        return results
