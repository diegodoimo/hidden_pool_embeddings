# Report: Aligning DDP evaluation with MTEB (`eval_model_with_ddp.py` vs `eval_model_with_mteb.py`)

## Problem

JSON summaries under `results/performace_evals/` diverged between runs driven by `eval_model_with_mteb.py` (prefix `mteb_`) and `eval_model_with_ddp.py` (custom encoder + `inference/test_retrieval_ddp_update.py`). Gaps were largest on **Reranking**, **STS**, **Summarization**, and **Clustering**; other types can still differ slightly (batch size, `torch.compile`, etc.).

## Root causes

### Reranking

MTEB’s `SearchEncoderWrapper` scores **only** documents listed in `top_ranked` per query. The old custom path always ran full-corpus `search()`, which is a different task definition.

`skip_first_result` from `AbsTaskRetrieval` was not passed into `calculate_retrieval_scores`.

**Category average vs “golden” MTEB JSON.** English v2 has **two** reranking tasks (`AskUbuntuDupQuestions` ~0.63, `MindSmallReranking` ~0.31); the official **Reranking** line is their mean (~0.47). If the custom run **skips** `MindSmallReranking`, the summary averages **only** AskUbuntu (~0.63), which looks ~15 points high even when per-task scores match MTEB. Skip happened because `prepare_datasets` called `_get_max_split_size_from_hub`, which **sums row counts across every HF config/split** on the dataset card; for reranking-style data that **overestimates** total size, so `total_rows > max_samples` (default `1_000_000`) dropped the task while `eval_model_with_mteb.py` still ran it.

**Scoring details (after the `top_ranked` path exists).** `SearchEncoderWrapper._rerank_documents` uses **`model.similarity`** (MTEB default: **`cos_sim`**, L2-normalized dot), then **`torch.topk(..., min(top_k, len(candidates)))`** so the run dict contains **only** those top-scoring candidates—not every candidate with a raw score. A path that used **`torch.matmul`** (unnormalized dot) on **all** candidates diverges from that definition when embedding norms vary or when `n_candidates > top_k` (e.g. 1000 cap from `max(k_values)`).

### STS (`mteb._evaluators.any_sts_evaluator.AnySTSEvaluator`)

MTEB encodes **column 1 for all rows**, then **column 2 for all rows**, and pairs rows by index. It supports **different prompt types** per column (`input1_prompt_type`, `input2_prompt_type`).

The old path merged both columns into one list, **deduplicated with `hash(text)`** (unsafe: collisions; also breaks asymmetric prompts when the same string appears in both roles), applied a **single** query-style prompt to every line, and indexed into one embedding matrix. That does not match MTEB’s two-pass encoding.

**Primary STS metrics.** `AnySTSEvaluator` passes **`similarity_scores`** from **`compute_pairwise_similarity`** (model **`similarity_pairwise`** if present, else pairwise cosine). `AbsTaskSTS._calculate_scores` uses those for **`pearson` / `spearman`** when non-null; if **`similarity_scores` is `None`**, it falls back to cosine-only. Omitting **`similarity_scores`** in the custom path therefore diverges from MTEB whenever the task’s **`main_score`** follows the model similarity path.

### Summarization (`mteb._evaluators.text.summarization_evaluator.SummarizationEvaluator`)

MTEB encodes **all human summaries** in order, then **all machine summaries** in order (no cross-list deduplication), then uses **`mteb.similarity_functions.cos_sim` / `dot_score`** (Torch, `F.normalize`) for max-over-human scores, and **`model.similarity`** per human line for the `pearson` / `spearman` metrics.

The old path deduplicated human+machine with **`hash(text)`**, used **NumPy** cosine/dot for max scores, and set **`pearson` / `spearman` from cosine correlations only** instead of the model similarity path. There was also a **`self.rank` bug** in a branch that should use **`rank`**.

### Clustering (`mteb.abstasks.clustering.AbsTaskClustering._evaluate_subset`)

After the same **fraction / count subsample** via `rng_state.sample`, MTEB encodes **every** remaining row; long inputs are handled by the **encoder (truncation)**, not by dropping rows from the dataset.

The old path used **`prepare_text_dataset` → `create_dataset`**, which **removed** any document whose prompt exceeded **`max_length` (8192)**. That changes which points are embedded relative to MTEB and shifts **v-measure** (often by a few points on long-text cluster tasks such as ArXiv).

## Code changes

### `inference/evaluate/eval_retrieval.py`

- `top_ranked` + reranking-only scoring path; `skip_first_result` on task payload; full-corpus `search()` for retrieval only; commented legacy reference at file end.
- Reranking scores: **`mteb.similarity_functions.cos_sim`** (aligns with `SearchEncoderWrapper` / default `model.similarity`), then **top-`min(top_k, n_candidates)`** docs in the run dict (same truncation as MTEB), not raw **`matmul`** over all candidates.

### `inference/evaluate/eval_sts.py`

- Two HF datasets (`texts1`, `texts2`) with prompts from **`_sts_prompt_for_task`** (maps MTEB prompt types to `utils.create_datasets.PromptType`).
- Length filtering: drop pairs where either side exceeds `max_length`, then rebuild datasets (with an extra narrowing pass if needed).
- **`evaluate_one_sts`**: two `encode_dataset` calls; **sklearn** `paired_*` distances as in MTEB’s STS evaluator; **`similarity_scores`** from **`mteb.similarity_functions.compute_pairwise_similarity`** on the DDP-unwrapped module (same role as `AnySTSEvaluator`).
- **Legacy** hash/dedup/single-dataset flow commented at file end.

### `inference/evaluate/eval_summarization.py`

- Flatten human/machine in **row order** (no hash dedup); filter long/empty positions; optional second narrowing pass if a second filter appears.
- Two datasets: `texts_human`, `texts_machine`; two `encode_dataset` calls.
- Scoring loop uses **`cos_sim` / `dot_score`** from MTEB; **`pearson` / `spearman`** use **`_model_similarity_pair`** (`model.module.similarity` when present, else normalized dot product as fallback).
- **Legacy** hash/dedup path commented at file end.

### `inference/evaluate/eval_clustering.py`

- **`prepare_text_dataset(..., skip_length_filter=True)`** so subsampled rows are not dropped for length (only empty strings removed).
- **`make_collate_fn(..., truncation_max_length=CLUSTERING_TRUNCATION_MAX_LENGTH)`** with **`CLUSTERING_TRUNCATION_MAX_LENGTH = 8192`**: tokenizer uses **`truncation=True`** so long prompts match “keep row, truncate at encode” behavior.
- **Legacy** prepare/collate (filter-long, no truncation) commented inline.

### `utils/create_datasets.py`

- **`create_dataset(..., skip_length_filter=False)`**: when True, filter step only removes **empty** texts.

### `utils/dataloader_helpers.py`

- **`collate_fn_with_padding(..., truncation_max_length=None)`**: optional truncation before appending **`eot_id`**.

### `inference/evaluate/shared.py`

- **`prepare_text_dataset`**: `prompt_type`, **`skip_length_filter`**.
- **`encode_dataset`**: `prompt_type` passed through to `encode()`.
- **`make_collate_fn`**: optional **`truncation_max_length`**.

### `inference/test_retrieval_ddp_update.py`

- **`prepare_datasets`**: do **not** skip tasks on hub **`total_rows > max_samples`** when **`task_type == "Reranking"`**, so **`MindSmallReranking`** (and similar) are not dropped due to inflated hub metadata; Reranking **category** averages then match MTEB’s task set.

## Files touched

| File | Change |
|------|--------|
| `inference/evaluate/eval_retrieval.py` | Reranking (`cos_sim`, top-k cap) + `skip_first_result` + legacy comments |
| `inference/evaluate/eval_sts.py` | Two-pass STS + `compute_pairwise_similarity` + legacy comments |
| `inference/evaluate/eval_summarization.py` | MTEB-aligned summarization + legacy comments |
| `inference/evaluate/eval_clustering.py` | Truncate-at-collate + skip length filter + legacy comments |
| `inference/evaluate/shared.py` | prepare / encode / make_collate extensions |
| `inference/test_retrieval_ddp_update.py` | Reranking: no hub size skip; full benchmark coverage |
| `utils/create_datasets.py` | `skip_length_filter` on `create_dataset` |
| `utils/dataloader_helpers.py` | `truncation_max_length` on collate |
| `report.md` | This summary |

## What may still differ

- Default **batch size** (e.g. 32 vs 16), **`torch.compile`**, and small numeric drift.
- **Summarization** vs MTEB: gaps tend to be smaller than Reranking/STS; remaining drift may come from length filtering of human/machine lines, skipped samples (constant-score guards), or **`model.similarity`** vs fallback.
- **`model.similarity`**: summarization **pearson/spearman** may differ if the custom module has no **`similarity`** (fallback uses normalized dot).
- **Clustering truncation limit**: we use **8192** tokens (minus one when **`eot_id`** is appended). MTEB’s official wrapper may use another **model max**; adjust **`CLUSTERING_TRUNCATION_MAX_LENGTH`** if needed to match the HF model card.
- Non-clustering tasks still **drop** over-long rows in **`create_dataset`** unless they opt into **`skip_length_filter`** + collate truncation.

## Verification

Re-run `eval_model_with_ddp.py` on `MTEB(eng, v2)` and compare **Reranking**, **STS**, **Summarization**, and **Clustering** to the `mteb_*` JSONs. Confirm the custom **`results` JSON lists both reranking tasks** (`AskUbuntuDupQuestions`, `MindSmallReranking`) before comparing the **Reranking** category average.
