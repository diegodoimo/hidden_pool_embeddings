import torch
import numpy as np

from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    build_index_remap,
    EvalContext,
)


def _prepare_bitext_mining(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    pairs = task._get_pairs(task.parallel_subsets)

    if task.parallel_subsets:
        subset_list = [(task.dataset[eval_split], "parallel")]
    else:
        subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        col1, col2 = pairs[0]
        sentence1 = list(data_split[col1])
        sentence2 = list(data_split[col2])

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
            print(f"  [{hf_subset}] {len(sentence1)} sentence pairs for bitext mining")

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
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }


@torch.inference_mode()
def evaluate_one_bitext_mining(task_data, model, batch_size, eval_context: EvalContext):
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

    norms1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    emb1_norm = emb1 / norms1
    emb2_norm = emb2 / norms2

    nearest_neighbors = []
    chunk_size = 1000
    for start in range(0, len(emb1_norm), chunk_size):
        end = min(start + chunk_size, len(emb1_norm))
        sim = emb1_norm[start:end] @ emb2_norm.T
        top_idx = np.argmax(sim, axis=1)
        top_scores = sim[np.arange(len(top_idx)), top_idx]
        for idx, score in zip(top_idx, top_scores):
            nearest_neighbors.append({"corpus_id": int(idx), "score": float(score)})

    gold = list(zip(range(len(emb1)), range(len(emb1))))
    scores = task_obj._compute_metrics(nearest_neighbors, gold)

    return {main_score: scores[main_score]}
