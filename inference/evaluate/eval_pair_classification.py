import torch
import numpy as np
from datasets import Dataset as HFDataset
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from mteb._evaluators.pair_classification_evaluator import PairClassificationDistances

from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    build_index_remap,
    EvalContext,
)


def _prepare_pair_classification(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        if task.metadata.modalities == ["text"]:
            if isinstance(data_split, HFDataset) and len(data_split) == 1:
                data_split = data_split[0]

        input1_col = task.input1_column_name
        input2_col = task.input2_column_name
        label_col = task.label_column_name

        sentence1 = list(data_split[input1_col])
        sentence2 = list(data_split[input2_col])
        labels = list(data_split[label_col])

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
            labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]
            if rank == 0:
                n_removed = sum(1 for v in valid_mask if not v)
                print(
                    f"  [{hf_subset}] {n_removed} pairs removed due to filtered texts"
                )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "texts": texts_ds,
                "indices1": indices1,
                "indices2": indices2,
                "labels": labels,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }


@torch.inference_mode()
def evaluate_one_pair_classification(task_data, model, batch_size, eval_context: EvalContext):
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
    manhattan_distances_ = paired_manhattan_distances(emb1, emb2)
    euclidean_distances_ = paired_euclidean_distances(emb1, emb2)
    dot_scores = np.sum(emb1 * emb2, axis=1)

    distances = PairClassificationDistances(
        cosine_scores=cosine_scores.tolist(),
        euclidean_distances=euclidean_distances_.tolist(),
        manhattan_distances=manhattan_distances_.tolist(),
        similarity_scores=cosine_scores.tolist(),
        dot_scores=dot_scores.tolist(),
    )

    scores = task_obj._compute_metrics(distances, dataset["labels"])

    return {main_score: scores[main_score]}
