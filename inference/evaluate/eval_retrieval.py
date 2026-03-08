import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType
from mteb._evaluators.retrieval_metrics import (
    calculate_retrieval_scores,
    make_score_dict,
)

from inference.helpers import abs_task_preprocessing, search, encode
from inference.hard_negative_mining import estimate_chunk_sizes
from utils.create_datasets import create_dataset
from utils.dataloader_helpers import LenghtSortedSampler

from inference.evaluate.shared import make_collate_fn, EvalContext


def _prepare_retrieval(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    task.convert_v1_dataset_format_to_v2(num_proc=None)
    subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        data_split["relevant_docs"], data_split["queries"] = (
            _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )
        )

        queries_dataset = create_dataset(
            dataset=data_split["queries"],
            task_metadata=task.metadata,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            prompt_type=PromptType.query,
            max_length=4096,
        )

        corpus_dataset = create_dataset(
            dataset=data_split["corpus"],
            task_metadata=task.metadata,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            prompt_type=PromptType.document,
            max_length=4096,
        )

        relevant_docs = data_split["relevant_docs"]
        removed_query_ids = (
            set(queries_dataset.removed_ids) if queries_dataset.removed_ids else set()
        )
        removed_corpus_ids = (
            set(corpus_dataset.removed_ids) if corpus_dataset.removed_ids else set()
        )
        if removed_query_ids or removed_corpus_ids:
            if rank == 0:
                print(
                    f"  [{hf_subset}] filtered {len(removed_query_ids)} queries, "
                    f"{len(removed_corpus_ids)} corpus docs (>4096 tokens or empty)"
                )
            relevant_docs = {
                qid: {
                    did: s for did, s in docs.items() if did not in removed_corpus_ids
                }
                for qid, docs in relevant_docs.items()
                if qid not in removed_query_ids
            }
            relevant_docs = {qid: docs for qid, docs in relevant_docs.items() if docs}

        valid_qids = set(relevant_docs.keys())
        num_queries_lost = len(queries_dataset) - sum(
            1 for qid in queries_dataset["id"] if qid in valid_qids
        )
        if num_queries_lost > 0:
            if rank == 0:
                print(
                    f"  [{hf_subset}] Note: {num_queries_lost} valid queries were excluded "
                    f"because all their paired positives were removed"
                )
            queries_dataset = queries_dataset.filter(
                lambda batch: [qid in valid_qids for qid in batch["id"]],
                batched=True,
            )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "queries": queries_dataset,
                "corpus": corpus_dataset,
                "relevant_docs": relevant_docs,
            },
            "task_specific_scores": task.task_specific_scores,
            "ignore_identical_ids": task.ignore_identical_ids,
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
        }


def evaluate_one(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    top_k=None,
    k_values=[1, 3, 5, 10, 20, 100, 1000],
    skip_first_result: bool = False,
):
    dataset = task_data["dataset"]
    task_specific_scores = task_data["task_specific_scores"]
    ignore_identical_ids = task_data["ignore_identical_ids"]
    hf_split = task_data["hf_split"]
    hf_subset = task_data["hf_subset"]
    main_score = task_data["main_score"]

    if top_k is None:
        top_k = max(k_values)
    model.eval()

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )

    query_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["queries"]["id"])}
    doc_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["corpus"]["id"])}

    sampler_queries = None
    if eval_context.world_size > 1:
        sampler_queries = torch.utils.data.distributed.DistributedSampler(
            dataset["queries"], shuffle=False, drop_last=False
        )

    if eval_context.new_inference_mode:
        sampler_queries = LenghtSortedSampler(dataset["queries"])

    queries_loader = DataLoader(
        dataset["queries"],
        sampler=sampler_queries,
        batch_size=batch_size,
        num_workers=max(1, len(os.sched_getaffinity(0)) // 2 - 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dist.barrier()
    start = time.time()
    query_embeddings = encode(
        model,
        queries_loader,
        prompt_type=PromptType.query,
        world_size=eval_context.world_size,
    )
    dist.barrier()

    chunk_size, query_chunk_size = estimate_chunk_sizes(query_embeddings)

    _cs = torch.tensor(
        [chunk_size, query_chunk_size],
        dtype=torch.long,
        device=f"cuda:{eval_context.rank}",
    )
    dist.broadcast(_cs, src=0)
    chunk_size, query_chunk_size = int(_cs[0].item()), int(_cs[1].item())
    del _cs

    top_scores, top_indices = search(
        model=model,
        query_embeddings=query_embeddings,
        corpus_dataset=dataset["corpus"],
        collate_fn=collate_fn,
        top_k=top_k,
        batch_size=batch_size,
        estract_positives=False,
        chunk_size=chunk_size,
    )

    dist.barrier()

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
