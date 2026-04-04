import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from mteb.similarity_functions import cos_sim, dot_score

from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)


def _prepare_summarization(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)
    max_length = 8192

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

        # MTEB SummarizationEvaluator: flatten in row order — no hash-based deduplication.
        human_flat = [s for hs in human_summaries for s in hs]
        machine_flat = [s for ms in machine_summaries for s in ms]

        if rank == 0:
            print(
                f"  [{hf_subset}] {len(human_summaries)} samples, "
                f"{len(human_flat)} human summaries, "
                f"{len(machine_flat)} machine summaries"
            )

        _, rem_h = prepare_text_dataset(
            human_flat,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
        )
        _, rem_m = prepare_text_dataset(
            machine_flat,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
        )

        new_human_flat = []
        new_machine_flat = []
        new_human_lens = []
        new_machine_lens = []
        new_gold_scores = []
        h_off, m_off = 0, 0
        n_samples_orig = len(human_lens)
        for i in range(n_samples_orig):
            h_len = human_lens[i]
            m_len = machine_lens[i]
            kept_h_idx = [h_off + j for j in range(h_len) if (h_off + j) not in rem_h]
            kept_m = [
                (m_off + j, normalized_scores[i][j])
                for j in range(m_len)
                if (m_off + j) not in rem_m
            ]
            h_off += h_len
            m_off += m_len
            if kept_h_idx and kept_m:
                new_human_flat.extend(human_flat[j] for j in kept_h_idx)
                new_machine_flat.extend(machine_flat[idx] for idx, _ in kept_m)
                new_human_lens.append(len(kept_h_idx))
                new_machine_lens.append(len(kept_m))
                new_gold_scores.append([s for _, s in kept_m])

        if rank == 0 and (len(new_human_lens) < n_samples_orig or rem_h or rem_m):
            print(
                f"  [{hf_subset}] summarization: dropped partial/long texts; "
                f"{n_samples_orig - len(new_human_lens)} samples removed entirely"
            )

        texts_human_ds, rem_h2 = prepare_text_dataset(
            new_human_flat,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
        )
        texts_machine_ds, rem_m2 = prepare_text_dataset(
            new_machine_flat,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=max_length,
        )
        if rem_h2 or rem_m2:
            nh2 = []
            nm2 = []
            hl2 = []
            ml2 = []
            gs2 = []
            ho, mo = 0, 0
            for i in range(len(new_human_lens)):
                hl = new_human_lens[i]
                ml = new_machine_lens[i]
                kh = [ho + j for j in range(hl) if (ho + j) not in rem_h2]
                km = [
                    (mo + j, new_gold_scores[i][j])
                    for j in range(ml)
                    if (mo + j) not in rem_m2
                ]
                ho += hl
                mo += ml
                if kh and km:
                    nh2.extend(new_human_flat[j] for j in kh)
                    nm2.extend(new_machine_flat[idx] for idx, _ in km)
                    hl2.append(len(kh))
                    ml2.append(len(km))
                    gs2.append([s for _, s in km])
            new_human_flat, new_machine_flat = nh2, nm2
            new_human_lens, new_machine_lens = hl2, ml2
            new_gold_scores = gs2
            texts_human_ds, _ = prepare_text_dataset(
                new_human_flat,
                task.metadata,
                instruction_template,
                tokenizer,
                rank,
                max_length=max_length,
            )
            texts_machine_ds, _ = prepare_text_dataset(
                new_machine_flat,
                task.metadata,
                instruction_template,
                tokenizer,
                rank,
                max_length=max_length,
            )

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
            "dataset": {
                "texts_human": texts_human_ds,
                "texts_machine": texts_machine_ds,
                "human_lens": new_human_lens,
                "machine_lens": new_machine_lens,
                "gold_scores": new_gold_scores,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }


def _model_similarity_pair(model, a: torch.Tensor, b: torch.Tensor) -> float:
    """Best-effort model.similarity for two 1D embedding vectors (handles DDP wrap)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "similarity") and callable(m.similarity):
        return float(m.similarity(a, b))
    return float(
        torch.dot(
            torch.nn.functional.normalize(a, dim=-1),
            torch.nn.functional.normalize(b, dim=-1),
        )
    )


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
    emb_human = encode_dataset(model, dataset["texts_human"], batch_size, collate_fn)
    emb_machine = encode_dataset(
        model, dataset["texts_machine"], batch_size, collate_fn
    )

    emb_human = emb_human.cpu().float()
    emb_machine = emb_machine.cpu().float()

    human_lens = dataset["human_lens"]
    machine_lens = dataset["machine_lens"]
    gold_scores = dataset["gold_scores"]

    embs_human_split = torch.split(emb_human, human_lens, dim=0)
    embs_machine_split = torch.split(emb_machine, machine_lens, dim=0)

    cosine_spearman_scores = []
    cosine_pearson_scores = []
    dot_spearman_scores = []
    dot_pearson_scores = []
    sim_spearman_scores = []
    sim_pearson_scores = []

    for embs_human_i, embs_machine_i, gold_row in zip(
        embs_human_split, embs_machine_split, gold_scores
    ):
        cosine_pred = []
        dot_pred = []
        sim_pred = []
        human_scores = []

        for emb_machine, gold_score in zip(embs_machine_i, gold_row):
            emb_m = emb_machine
            emb_h_block = embs_human_i

            cosine_scores = cos_sim(emb_m, emb_h_block)
            dot_scores = dot_score(emb_m, emb_h_block)
            cosine_max_score = float(torch.max(cosine_scores).item())
            dot_max_score = float(torch.max(dot_scores).item())

            sim_max_score = max(
                _model_similarity_pair(model, emb_m, emb_h_block[j])
                for j in range(emb_h_block.shape[0])
            )

            cosine_pred.append(cosine_max_score)
            dot_pred.append(dot_max_score)
            sim_pred.append(sim_max_score)
            human_scores.append(gold_score)

        if (
            len(set(human_scores)) == 1
            or len(set(dot_pred)) == 1
            or len(set(cosine_pred)) == 1
        ):
            continue

        cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred).statistic)
        cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred).statistic)
        dot_spearman_scores.append(spearmanr(human_scores, dot_pred).statistic)
        dot_pearson_scores.append(pearsonr(human_scores, dot_pred).statistic)
        sim_spearman_scores.append(spearmanr(human_scores, sim_pred).statistic)
        sim_pearson_scores.append(pearsonr(human_scores, sim_pred).statistic)

    scores = {
        "cosine_spearman": float(np.mean(cosine_spearman_scores)),
        "cosine_pearson": float(np.mean(cosine_pearson_scores)),
        "dot_spearman": float(np.mean(dot_spearman_scores)),
        "dot_pearson": float(np.mean(dot_pearson_scores)),
        "pearson": float(np.mean(sim_pearson_scores)),
        "spearman": float(np.mean(sim_spearman_scores)),
    }

    return {main_score: scores[main_score]}


@torch.inference_mode()
def evaluate_hidden_states_summarization(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    target_layers,
    use_last_token=False,
):
    model.eval()
    dataset = task_data["dataset"]
    main_score = task_data["main_score"]

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    human_embs_dict = encode_dataset(
        model,
        dataset["texts_human"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )
    machine_embs_dict = encode_dataset(
        model,
        dataset["texts_machine"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
    )

    human_lens = dataset["human_lens"]
    machine_lens = dataset["machine_lens"]
    gold_scores = dataset["gold_scores"]

    layer_performance = {}
    for (layer, emb_human), (_, emb_machine) in zip(
        human_embs_dict.items(), machine_embs_dict.items()
    ):
        embs_human_split = torch.split(emb_human.float(), human_lens, dim=0)
        embs_machine_split = torch.split(emb_machine.float(), machine_lens, dim=0)

        cosine_spearman_scores = []
        cosine_pearson_scores = []

        for embs_human_i, embs_machine_i, gold_row in zip(
            embs_human_split, embs_machine_split, gold_scores
        ):
            cosine_pred = []
            human_scores_row = []
            for emb_m, gold_score in zip(embs_machine_i, gold_row):
                cosine_max = float(torch.max(cos_sim(emb_m, embs_human_i)).item())
                cosine_pred.append(cosine_max)
                human_scores_row.append(gold_score)

            if len(set(human_scores_row)) == 1 or len(set(cosine_pred)) == 1:
                continue

            cosine_spearman_scores.append(
                spearmanr(human_scores_row, cosine_pred).statistic
            )
            cosine_pearson_scores.append(
                pearsonr(human_scores_row, cosine_pred).statistic
            )

        scores = {
            "cosine_spearman": float(np.mean(cosine_spearman_scores)),
            "cosine_pearson": float(np.mean(cosine_pearson_scores)),
            "dot_spearman": float(np.mean(cosine_spearman_scores)),
            "dot_pearson": float(np.mean(cosine_pearson_scores)),
            "pearson": float(np.mean(cosine_pearson_scores)),
            "spearman": float(np.mean(cosine_spearman_scores)),
        }
        layer_performance[layer] = scores[main_score]

    return layer_performance


# Legacy (pre–MTEB alignment): merged human+machine texts into one list, deduplicated
# with hash(text), single encode_dataset on dataset["texts"], and numpy cosine/dot
# max — diverged from SummarizationEvaluator (separate encodes, mteb cos_sim/dot_score,
# model.similarity for pearson/spearman).
#
#         all_texts = all_human + all_machine
#         unique_texts, text_to_idx = [], {}
#         for text in all_texts:
#             h = hash(text)
#             ...
#         texts_ds, removed = prepare_text_dataset(unique_texts, ...)
#         # bug: if removed: ... used self.rank instead of rank
