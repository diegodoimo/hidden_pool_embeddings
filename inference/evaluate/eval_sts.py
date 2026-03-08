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
    build_index_remap,
    EvalContext,
)


def _prepare_sts(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        col1, col2 = task.column_names

        sentence1 = list(data_split[col1])
        sentence2 = list(data_split[col2])
        raw_scores = list(data_split["score"])
        normalized_scores = [task._normalize(s) for s in raw_scores]

        all_sentences = sentence1 + sentence2
        unique_texts, text_to_idx = [], {}
        for text in all_sentences:
            h = hash(text)
            if h not in text_to_idx:
                text_to_idx[h] = len(unique_texts)
                unique_texts.append(text)

        indices1 = [text_to_idx[hash(s)] for s in sentence1]
        indices2 = [text_to_idx[hash(s)] for s in sentence2]

        if rank == 0:
            n_dedup = len(all_sentences) - len(unique_texts)
            print(
                f"  [{hf_subset}] {n_dedup}/{len(all_sentences)} duplicate texts deduplicated"
            )

        texts_ds, removed = prepare_text_dataset(
            unique_texts, task.metadata, instruction_template, tokenizer, rank
        )

        if removed:
            old_to_new = build_index_remap(len(unique_texts), removed)
            valid_mask = [
                indices1[i] not in removed and indices2[i] not in removed
                for i in range(len(indices1))
            ]
            indices1 = [
                old_to_new[indices1[i]] for i in range(len(indices1)) if valid_mask[i]
            ]
            indices2 = [
                old_to_new[indices2[i]] for i in range(len(indices2)) if valid_mask[i]
            ]
            normalized_scores = [
                normalized_scores[i]
                for i in range(len(normalized_scores))
                if valid_mask[i]
            ]
            if rank == 0:
                n_removed = sum(1 for v in valid_mask if not v)
                print(
                    f"  [{hf_subset}] {n_removed} STS pairs removed due to filtered texts"
                )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "texts": texts_ds,
                "indices1": indices1,
                "indices2": indices2,
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
    embeddings = encode_dataset(model, dataset["texts"], batch_size, collate_fn)
    embeddings_np = embeddings.cpu().numpy()

    emb1 = embeddings_np[dataset["indices1"]]
    emb2 = embeddings_np[dataset["indices2"]]

    cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
    manhattan_distances_ = -paired_manhattan_distances(emb1, emb2)
    euclidean_distances_ = -paired_euclidean_distances(emb1, emb2)

    scores_dict = {
        "cosine_scores": cosine_scores.tolist(),
        "manhattan_distances": manhattan_distances_.tolist(),
        "euclidean_distances": euclidean_distances_.tolist(),
        "similarity_scores": None,
    }

    scores = task_obj._calculate_scores(scores_dict, dataset["labels"])

    return {main_score: scores[main_score]}
