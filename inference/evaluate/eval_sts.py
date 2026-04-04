import torch
import numpy as np
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)
from mteb.types import PromptType as MtebPromptType
from mteb.similarity_functions import compute_pairwise_similarity
from utils.create_datasets import PromptType as LocalPromptType


def _sts_prompt_for_task(task, column: str) -> LocalPromptType:
    """Map MTEB STS column prompt types to local PromptType (create_datasets enum).

    MTEB's ``InstructSentenceTransformerModel`` (used for Qwen3) applies the
    instruction prefix whenever ``prompt_type`` is not ``PromptType.document``.
    The STS evaluator passes ``prompt_type=None`` (= ``input1/2_prompt_type``),
    which means instructions ARE applied.  We mirror this by returning
    ``query`` (which adds the ``Instruct: …\\nQuery:`` prefix).
    """
    if column == "first":
        p = getattr(task, "input1_prompt_type", None)
    else:
        p = getattr(task, "input2_prompt_type", None)
    if p is None:
        return LocalPromptType.query
    if p == MtebPromptType.query or getattr(p, "value", None) == "query":
        return LocalPromptType.query
    return LocalPromptType.document


def _removed_too_long_sets(
    sentence1,
    sentence2,
    task_metadata,
    instruction_template,
    tokenizer,
    rank,
    max_length,
    p1,
    p2,
):
    """Probe length filtering per column (matches prepare_text_dataset / create_dataset)."""
    _, rem1 = prepare_text_dataset(
        sentence1,
        task_metadata,
        instruction_template,
        tokenizer,
        rank,
        max_length=max_length,
        prompt_type=p1,
    )
    _, rem2 = prepare_text_dataset(
        sentence2,
        task_metadata,
        instruction_template,
        tokenizer,
        rank,
        max_length=max_length,
        prompt_type=p2,
    )
    return rem1, rem2


def _prepare_sts(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)
    p1 = _sts_prompt_for_task(task, "first")
    p2 = _sts_prompt_for_task(task, "second")
    max_length = 8192

    for data_split, hf_subset in subset_list:
        col1, col2 = task.column_names

        sentence1 = list(data_split[col1])
        sentence2 = list(data_split[col2])
        raw_scores = list(data_split["score"])
        normalized_scores = [task._normalize(s) for s in raw_scores]

        # Align with mteb._evaluators.any_sts_evaluator.AnySTSEvaluator: encode each column
        # separately in row order (no cross-column hash deduplication).
        rem1, rem2 = _removed_too_long_sets(
            sentence1,
            sentence2,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length,
            p1,
            p2,
        )
        if rem1 or rem2:
            valid = [
                i for i in range(len(sentence1)) if i not in rem1 and i not in rem2
            ]
            if rank == 0:
                print(
                    f"  [{hf_subset}] {len(sentence1) - len(valid)} STS pairs removed "
                    f"(>{max_length} tokens or empty in either column)"
                )
            sentence1 = [sentence1[i] for i in valid]
            sentence2 = [sentence2[i] for i in valid]
            normalized_scores = [normalized_scores[i] for i in valid]

        texts1_ds, rem1b = prepare_text_dataset(
            sentence1,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
            prompt_type=p1,
        )
        texts2_ds, rem2b = prepare_text_dataset(
            sentence2,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
            prompt_type=p2,
        )
        if rem1b or rem2b:
            valid = [
                i for i in range(len(sentence1)) if i not in rem1b and i not in rem2b
            ]
            sentence1 = [sentence1[i] for i in valid]
            sentence2 = [sentence2[i] for i in valid]
            normalized_scores = [normalized_scores[i] for i in valid]
            texts1_ds, _ = prepare_text_dataset(
                sentence1,
                task.metadata,
                instruction_template,
                tokenizer,
                rank,
                max_length=max_length,
                prompt_type=p1,
            )
            texts2_ds, _ = prepare_text_dataset(
                sentence2,
                task.metadata,
                instruction_template,
                tokenizer,
                rank,
                max_length=max_length,
                prompt_type=p2,
            )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "texts1": texts1_ds,
                "texts2": texts2_ds,
                "labels": normalized_scores,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }


@torch.inference_mode()
def evaluate_one_sts(task_data, model, batch_size, eval_context: EvalContext):
    model.eval()
    dataset = task_data["dataset"]
    task_obj = task_data["task_obj"]
    main_score = task_data["main_score"]

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    embeddings1 = encode_dataset(model, dataset["texts1"], batch_size, collate_fn)
    embeddings2 = encode_dataset(model, dataset["texts2"], batch_size, collate_fn)
    embeddings1 = embeddings1.cpu().numpy()
    embeddings2 = embeddings2.cpu().numpy()

    cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    manhattan_distances_ = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances_ = -paired_euclidean_distances(embeddings1, embeddings2)

    m = model.module if hasattr(model, "module") else model
    sim = compute_pairwise_similarity(m, embeddings1, embeddings2)
    if hasattr(sim, "tolist"):
        similarity_list = sim.tolist()
    else:
        similarity_list = list(sim)

    scores_dict = {
        "cosine_scores": cosine_scores.tolist(),
        "manhattan_distances": manhattan_distances_.tolist(),
        "euclidean_distances": euclidean_distances_.tolist(),
        "similarity_scores": similarity_list,
    }

    scores = task_obj._calculate_scores(scores_dict, dataset["labels"])

    return {main_score: scores[main_score]}


@torch.inference_mode()
def evaluate_hidden_states_sts(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    target_layers,
    use_last_token=False,
):
    model.eval()
    dataset = task_data["dataset"]
    task_obj = task_data["task_obj"]
    main_score = task_data["main_score"]

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    embeddings1_dict = encode_dataset(
        model,
        dataset["texts1"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )
    embeddings2_dict = encode_dataset(
        model,
        dataset["texts2"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )

    layer_performance = {}
    for (layer, emb1), (_, emb2) in zip(
        embeddings1_dict.items(), embeddings2_dict.items()
    ):
        emb1_np = emb1.numpy()
        emb2_np = emb2.numpy()

        cosine_scores = 1 - paired_cosine_distances(emb1_np, emb2_np)
        manhattan_distances_ = -paired_manhattan_distances(emb1_np, emb2_np)
        euclidean_distances_ = -paired_euclidean_distances(emb1_np, emb2_np)

        scores_dict = {
            "cosine_scores": cosine_scores.tolist(),
            "manhattan_distances": manhattan_distances_.tolist(),
            "euclidean_distances": euclidean_distances_.tolist(),
            "similarity_scores": cosine_scores.tolist(),
        }
        scores = task_obj._calculate_scores(scores_dict, dataset["labels"])
        layer_performance[layer] = scores[main_score]

    return layer_performance


# Legacy (pre–MTEB alignment): single pooled dataset of unique texts across both STS
# columns, keyed by hash(text) (collision-prone) and one PromptType.query for every
# line — wrong for asymmetric STS and not equivalent to AnySTSEvaluator's two passes.
#
#     all_sentences = sentence1 + sentence2
#     unique_texts, text_to_idx = [], {}
#     for text in all_sentences:
#         h = hash(text)
#         if h not in text_to_idx:
#             text_to_idx[h] = len(unique_texts)
#             unique_texts.append(text)
#     indices1 = [text_to_idx[hash(s)] for s in sentence1]
#     indices2 = [text_to_idx[hash(s)] for s in sentence2]
#     texts_ds, removed = prepare_text_dataset(unique_texts, ..., prompt_type=query)
#     ...
#     embeddings = encode_dataset(model, dataset["texts"], ...)
#     emb1 = embeddings_np[dataset["indices1"]]
#     emb2 = embeddings_np[dataset["indices2"]]
