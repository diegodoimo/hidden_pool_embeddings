import torch
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType
from mteb.similarity_functions import cos_sim
from mteb._evaluators.retrieval_metrics import (
    calculate_retrieval_scores,
    make_score_dict,
)

from inference.helpers import abs_task_preprocessing
from utils.create_datasets import create_dataset
from inference.evaluate.shared import make_collate_fn, EvalContext, encode_dataset


def _prepare_reranking(
    task,
    task_name,
    eval_split,
    instruction_template,
    datasets,
    tokenizer,
    rank,
    max_eval_queries=None,
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

        top_ranked = data_split.get("top_ranked")
        if top_ranked:
            top_ranked = {
                qid: [d for d in ids if d not in removed_corpus_ids]
                for qid, ids in top_ranked.items()
                if qid not in removed_query_ids
            }
            top_ranked = {k: v for k, v in top_ranked.items() if v}
            top_ranked = {k: v for k, v in top_ranked.items() if k in valid_qids}

        if (
            max_eval_queries is not None
            and task_name == "MindSmallReranking"
            and len(queries_dataset) > max_eval_queries
        ):
            import random as _rng

            all_qids = list(queries_dataset["id"])
            sampled_qids = set(_rng.Random(42).sample(all_qids, max_eval_queries))
            if rank == 0:
                print(
                    f"  [{task_name}] subsampling queries: "
                    f"{len(all_qids)} -> {max_eval_queries}"
                )
            queries_dataset = queries_dataset.filter(
                lambda batch: [qid in sampled_qids for qid in batch["id"]],
                batched=True,
            )
            relevant_docs = {
                qid: docs for qid, docs in relevant_docs.items() if qid in sampled_qids
            }
            if top_ranked is not None:
                top_ranked = {
                    qid: docs for qid, docs in top_ranked.items() if qid in sampled_qids
                }

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "queries": queries_dataset,
                "corpus": corpus_dataset,
                "relevant_docs": relevant_docs,
            },
            "task_specific_scores": task.task_specific_scores,
            "ignore_identical_ids": task.ignore_identical_ids,
            "skip_first_result": getattr(task, "skip_first_result", False),
            "top_ranked": top_ranked if top_ranked else None,
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
        }


def evaluate_one_reranking(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    top_k=None,
    k_values=None,
):
    """Score only pre-ranked candidate docs per query (MTEB reranking protocol).

    Matches mteb.models.search_wrappers.SearchEncoderWrapper._rerank_documents:
    uses model.similarity (cos_sim) over the candidate set only, not a full-corpus search.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 100, 1000]
    dataset = task_data["dataset"]
    task_specific_scores = task_data["task_specific_scores"]
    ignore_identical_ids = task_data["ignore_identical_ids"]
    skip_first_result = task_data.get("skip_first_result", False)
    top_ranked = task_data.get("top_ranked")
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

    from inference.helpers import encode
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    import os
    from utils.dataloader_helpers import LenghtSortedSampler

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
    query_embeddings = encode(
        model,
        queries_loader,
        prompt_type=PromptType.query,
        world_size=eval_context.world_size,
    )
    dist.barrier()

    corpus_embeddings = encode_dataset(model, dataset["corpus"], batch_size, collate_fn)
    id_to_row = {did: i for i, did in enumerate(dataset["corpus"]["id"])}

    results = {}
    for q_idx, qid in query_idx_to_id.items():
        ranked_ids = top_ranked.get(qid) if top_ranked else None
        if not ranked_ids:
            results[qid] = {}
            continue
        rows = [id_to_row[d] for d in ranked_ids]
        doc_embs = corpus_embeddings[rows]
        qemb = query_embeddings[q_idx : q_idx + 1].to(doc_embs.dtype)
        scores_2d = cos_sim(qemb, doc_embs.to(qemb.device))
        n_cand = len(ranked_ids)
        k_take = min(top_k, n_cand)
        vals, doc_positions = torch.topk(scores_2d, k=k_take, dim=1, largest=True)
        results[qid] = {
            ranked_ids[int(doc_positions[0, j].item())]: vals[0, j].item()
            for j in range(k_take)
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


@torch.inference_mode()
def evaluate_hidden_states_reranking(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    target_layers,
    use_last_token=False,
    k_values=None,
    top_k=None,
):
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 100, 1000]
    if top_k is None:
        top_k = max(k_values)

    dataset = task_data["dataset"]
    task_specific_scores = task_data["task_specific_scores"]
    ignore_identical_ids = task_data["ignore_identical_ids"]
    skip_first_result = task_data.get("skip_first_result", False)
    top_ranked = task_data.get("top_ranked")
    hf_split = task_data["hf_split"]
    hf_subset = task_data["hf_subset"]
    main_score = task_data["main_score"]
    model.eval()

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )

    query_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["queries"]["id"])}

    query_embeddings_dict = encode_dataset(
        model,
        dataset["queries"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )
    corpus_embeddings_dict = encode_dataset(
        model,
        dataset["corpus"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )

    id_to_row = {did: i for i, did in enumerate(dataset["corpus"]["id"])}

    layer_performance = {}
    for (layer, query_embs), (_, corpus_embs) in zip(
        query_embeddings_dict.items(), corpus_embeddings_dict.items()
    ):
        results = {}
        for q_idx, qid in query_idx_to_id.items():
            ranked_ids = top_ranked.get(qid) if top_ranked else None
            if not ranked_ids:
                results[qid] = {}
                continue
            rows = [id_to_row[d] for d in ranked_ids]
            doc_embs = corpus_embs[rows].float()
            qemb = query_embs[q_idx : q_idx + 1].float()
            scores_2d = cos_sim(qemb, doc_embs)
            n_cand = len(ranked_ids)
            k_take = min(top_k, n_cand)
            vals, doc_positions = torch.topk(scores_2d, k=k_take, dim=1, largest=True)
            results[qid] = {
                ranked_ids[int(doc_positions[0, j].item())]: vals[0, j].item()
                for j in range(k_take)
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
            results, qrels, list(k_values), skip_first_result
        )

        task_specific_scores_ = task_specific_scores(
            all_scores,
            dataset["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        scores_dict = make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores_,
            None,
        )
        layer_performance[layer] = scores_dict[main_score]

    return layer_performance
