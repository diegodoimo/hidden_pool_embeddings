import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    build_index_remap,
    EvalContext,
)


def _prepare_summarization(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        text_col = task.text_column_name
        human_col = task.human_summaries_column_name
        machine_col = task.machine_summaries_column_name
        relevance_col = task.relevancy_column_name

        human_summaries = list(data_split[human_col])
        machine_summaries = list(data_split[machine_col])
        relevance = list(data_split[relevance_col])

        normalized_scores = [
            (
                (np.array(x) - task.min_score) / (task.max_score - task.min_score)
            ).tolist()
            for x in relevance
        ]

        human_lens = [len(hs) for hs in human_summaries]
        machine_lens = [len(ms) for ms in machine_summaries]

        all_human = [s for hs in human_summaries for s in hs]
        all_machine = [s for ms in machine_summaries for s in ms]

        all_texts = all_human + all_machine
        unique_texts, text_to_idx = [], {}
        for text in all_texts:
            h = hash(text)
            if h not in text_to_idx:
                text_to_idx[h] = len(unique_texts)
                unique_texts.append(text)

        human_indices = [text_to_idx[hash(s)] for s in all_human]
        machine_indices = [text_to_idx[hash(s)] for s in all_machine]

        if rank == 0:
            n_dedup = len(all_texts) - len(unique_texts)
            print(
                f"  [{hf_subset}] {n_dedup}/{len(all_texts)} duplicate summaries deduplicated"
            )
            print(
                f"  [{hf_subset}] {len(human_summaries)} samples, "
                f"{sum(human_lens)} human summaries, "
                f"{sum(machine_lens)} machine summaries"
            )

        texts_ds, removed = prepare_text_dataset(
            unique_texts, task.metadata, instruction_template, tokenizer, rank
        )

        if removed:
            old_to_new = build_index_remap(len(unique_texts), removed)
            new_human_indices = []
            new_machine_indices = []
            new_human_lens = []
            new_machine_lens = []
            new_gold_scores = []
            h_off, m_off = 0, 0
            n_samples_orig = len(human_lens)
            for i in range(n_samples_orig):
                h_len = human_lens[i]
                m_len = machine_lens[i]
                s_h = human_indices[h_off : h_off + h_len]
                s_m = machine_indices[m_off : m_off + m_len]
                s_scores = normalized_scores[i]
                kept_h = [idx for idx in s_h if idx not in removed]
                kept_m_scores = [
                    (idx, s_scores[j])
                    for j, idx in enumerate(s_m)
                    if idx not in removed
                ]
                if kept_h and kept_m_scores:
                    new_human_indices.extend(old_to_new[idx] for idx in kept_h)
                    new_machine_indices.extend(
                        old_to_new[idx] for idx, _ in kept_m_scores
                    )
                    new_human_lens.append(len(kept_h))
                    new_machine_lens.append(len(kept_m_scores))
                    new_gold_scores.append([s for _, s in kept_m_scores])
                h_off += h_len
                m_off += m_len
            human_indices = new_human_indices
            machine_indices = new_machine_indices
            human_lens = new_human_lens
            machine_lens = new_machine_lens
            normalized_scores = new_gold_scores
            if self.rank == 0 and len(human_lens) < n_samples_orig:
                print(
                    f"  [{hf_subset}] {n_samples_orig - len(human_lens)} summarization samples "
                    f"removed due to filtered texts"
                )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "texts": texts_ds,
                "human_indices": human_indices,
                "machine_indices": machine_indices,
                "human_lens": human_lens,
                "machine_lens": machine_lens,
                "gold_scores": normalized_scores,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }


@torch.inference_mode()
def evaluate_one_summarization(task_data, model, batch_size, eval_context: EvalContext):
    model.eval()
    dataset = task_data["dataset"]
    main_score = task_data["main_score"]

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    embeddings = encode_dataset(model, dataset["texts"], batch_size, collate_fn)
    embeddings_np = embeddings.cpu().numpy()

    human_indices = dataset["human_indices"]
    machine_indices = dataset["machine_indices"]
    human_lens = dataset["human_lens"]
    machine_lens = dataset["machine_lens"]
    gold_scores = dataset["gold_scores"]

    embs_human = embeddings_np[human_indices]
    embs_machine = embeddings_np[machine_indices]

    embs_human_per_sample = np.split(embs_human, np.cumsum(human_lens)[:-1])
    embs_machine_per_sample = np.split(embs_machine, np.cumsum(machine_lens)[:-1])

    cosine_spearman_scores = []
    cosine_pearson_scores = []
    dot_spearman_scores = []
    dot_pearson_scores = []

    n_skipped = 0
    for i, (embs_human_i, embs_machine_i) in enumerate(
        zip(embs_human_per_sample, embs_machine_per_sample)
    ):
        cosine_pred = []
        dot_pred = []
        human_scores = []

        embs_human_i_norm = embs_human_i / np.linalg.norm(
            embs_human_i, axis=1, keepdims=True
        )

        for emb_machine, gold_score in zip(embs_machine_i, gold_scores[i]):
            emb_machine_norm = emb_machine / np.linalg.norm(emb_machine)
            cos_scores = emb_machine_norm @ embs_human_i_norm.T
            dot_scores = emb_machine @ embs_human_i.T

            cosine_pred.append(float(np.max(cos_scores)))
            dot_pred.append(float(np.max(dot_scores)))
            human_scores.append(gold_score)

        if (
            len(set(human_scores)) == 1
            or len(set(dot_pred)) == 1
            or len(set(cosine_pred)) == 1
        ):
            n_skipped += 1
            continue

        cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred).statistic)
        cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred).statistic)
        dot_spearman_scores.append(spearmanr(human_scores, dot_pred).statistic)
        dot_pearson_scores.append(pearsonr(human_scores, dot_pred).statistic)

    scores = {
        "cosine_spearman": float(np.mean(cosine_spearman_scores)),
        "cosine_pearson": float(np.mean(cosine_pearson_scores)),
        "dot_spearman": float(np.mean(dot_spearman_scores)),
        "dot_pearson": float(np.mean(dot_pearson_scores)),
        "pearson": float(np.mean(cosine_pearson_scores)),
        "spearman": float(np.mean(cosine_spearman_scores)),
    }

    return {main_score: scores[main_score]}
