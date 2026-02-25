# Task Type Extension Report

## Overview

Extended `test_retrieval_ddp_update.py` to support **PairClassification**, **MultilabelClassification**, and **Clustering** tasks alongside the existing **Retrieval** flow. The `prepare_datasets` / `evaluate` pipeline now dispatches by `task.metadata.type`, with a dedicated prepare + evaluate method pair per task type — mirroring the MTEB structure where each `AbsTask*` subclass has its own `_evaluate_subset`.

## Shared Helpers

- **`_prepare_text_dataset(texts, task_metadata, instruction_template)`** — wraps raw text lists into a prompt-augmented HF dataset via `create_dataset`, reusing the instruction template (embeddinggemma/qwen3). Uses `max_length=100_000` to effectively disable length filtering for non-retrieval tasks.
- **`_encode_dataset(model, dataset, batch_size)`** — DDP-aware encoding pipeline using `LenghtSortedSampler` + the `encode()` function from `inference.helpers` (length-sorted batching, all-gather, index reordering).

## PairClassification

- **`_prepare_pair_classification`** — loads `sentence1`/`sentence2`/`labels`, handles v1 single-row format, deduplicates texts (using `hash()`), stores index mappings.
- **`evaluate_one_pair_classification`** — encodes unique texts once, reconstructs pair embeddings via index lookup, computes cosine/euclidean/manhattan/dot distances, calls `task._compute_metrics()` from MTEB to get `max_ap`, `max_f1`, etc.

## MultilabelClassification

- **`_prepare_multilabel_classification`** — loads both train and test splits (classification needs both, unlike retrieval), subsamples test to 2000 with stratification (as MTEB does), prepares both datasets.
- **`evaluate_one_multilabel_classification`** — encodes train + test once, then runs `n_experiments` bootstrap experiments: undersample train via `task._undersample_data_indices()`, binarize labels with `MultiLabelBinarizer`, fit classifier via `_evaluate_classifier` from MTEB, score via `task._calculate_scores()`. Returns averaged main score.

## Clustering

- **`_prepare_clustering`** — loads `sentences`/`labels`, applies downsampling (`max_fraction_of_documents_to_embed` / `max_document_to_embed`) using `task.rng_state` for reproducibility.
- **`evaluate_one_clustering`** — encodes all texts, calls `_evaluate_clustering_bootstrapped` from MTEB (bootstrapped K-means + V-measure). Returns mean V-measure.

## `evaluate()` Dispatch

Updated to route by `task_type`: `"Retrieval"` calls `evaluate_one`, `"PairClassification"` calls `evaluate_one_pair_classification`, etc. Unsupported types are skipped with a warning. Results are still grouped by task type in the returned dict.

## MTEB Reuse

The implementation maximizes MTEB imports:
- `_evaluate_clustering_bootstrapped` (clustering scoring)
- `_evaluate_classifier` (multilabel classifier training)
- `PairClassificationDistances` (pair classification distance container)
- Task object methods: `_compute_metrics`, `_calculate_scores`, `_undersample_data_indices`

All encoding is routed through the existing DDP pipeline.

---

## Classification

- **`_prepare_classification`** — loads train and test splits using `task.input_column_name` / `task.label_column_name` (defaults: `"text"` / `"label"`). Follows the same subset-handling logic as MultilabelClassification (`cast`, `hf_subsets`, `select_columns`). No test-set subsampling (matches upstream MTEB behavior for single-label classification).
- **`evaluate_one_classification`** — encodes train + test once, then runs `n_experiments` bootstrap experiments. Each experiment replicates MTEB's `AbsTaskClassification._undersample_data` logic: shuffles indices with `np.random.RandomState(seed)` and picks `samples_per_label` samples per label. Fits a cloned `task.evaluator_model` (default: `LogisticRegression(n_jobs=-1, max_iter=100)`), predicts on test, delegates scoring to `task._calculate_scores(y_test, y_pred)` which returns accuracy, f1, precision, recall, ap. Averages across experiments, handling `None` values (ap is None for non-binary tasks).

## STS

- **`_prepare_sts`** — extracts sentence pairs from `task.column_names` (default: `("sentence1", "sentence2")`) and raw scores from `"score"` column. Normalizes scores via `task._normalize()` (`(x - min_score) / (max_score - min_score)`). Deduplicates all unique texts (same hash-based approach as PairClassification), stores index mappings for reconstruction.
- **`evaluate_one_sts`** — encodes unique texts once, reconstructs paired embeddings via index lookup, computes cosine similarity (`1 - paired_cosine_distances`), negated manhattan (`-paired_manhattan_distances`), and negated euclidean (`-paired_euclidean_distances`) distances. Sets `similarity_scores=None` (no custom model similarity function), which causes `_calculate_scores` to fall back to cosine. Delegates correlation computation to `task._calculate_scores(scores_dict, normalized_scores)` which returns pearson/spearman correlations for each distance metric.

## Reranking

- `AbsTaskReranking` inherits from `AbsTaskRetrieval` — after `load_data()` the data is already in retrieval format (corpus, queries, relevant_docs). No new preparation or evaluation methods needed; both `prepare_datasets` and `evaluate` dispatchers route `"Reranking"` directly to the existing `_prepare_retrieval` and `evaluate_one` methods.

## Updated `evaluate()` Dispatch

Now routes seven task types: `"Retrieval"`, `"Reranking"` (both to `evaluate_one`), `"PairClassification"`, `"MultilabelClassification"`, `"Clustering"`, `"Classification"`, and `"STS"`. Unsupported types are skipped with a warning.

## Additional MTEB Reuse (new)

- `AbsTaskClassification._calculate_scores` (accuracy, f1, precision, recall, ap)
- `AbsTaskSTS._calculate_scores` (pearson/spearman correlations)
- `AbsTaskSTS._normalize` (score normalization)
- `AbsTaskClassification.evaluator_model` / `samples_per_label` / `n_experiments` / `seed`
- `AbsTaskSTS.column_names` / `min_score` / `max_score`
- `sklearn.base.clone` (already imported) for cloning the evaluator model per experiment

---

## Summarization

- **`_prepare_summarization`** — uses `abs_task_preprocessing` to get the data split and subset. Reads columns via the task's own attributes: `task.text_column_name`, `task.human_summaries_column_name`, `task.machine_summaries_column_name`, `task.relevancy_column_name` (defined on `AbsTaskSummarization`). Normalizes relevance scores using `(x - task.min_score) / (task.max_score - task.min_score)` (matching `AbsTaskSummarization._evaluate_subset`). Flattens all human and machine summaries into a single list, deduplicates with the same hash-based approach used for STS/PairClassification, and encodes via `_prepare_text_dataset`. Stores per-sample lengths (`human_lens`, `machine_lens`) and flat index mappings (`human_indices`, `machine_indices`) so embeddings can be split back into per-sample groups at evaluation time.
- **`evaluate_one_summarization`** — encodes all unique summary texts once via the DDP-aware `_encode_dataset`, then reconstructs per-sample human/machine embedding arrays using `np.split` with cumulative lengths. Reproduces the `SummarizationEvaluator.__call__` + `_calculate_metrics` logic: for each sample, computes max cosine/dot similarity between each machine summary and all human summaries of that sample (predicted quality score), skips samples where all human scores, cosine predictions, or dot predictions are constant (matching the evaluator's guard), then computes Pearson and Spearman correlations between predicted and gold scores. Returns `cosine_spearman`, `cosine_pearson`, `dot_spearman`, `dot_pearson`, plus `pearson`/`spearman` aliases (cosine-based, since the pipeline's embeddings are L2-normalized so cosine ≡ dot). The `main_score` (typically `cosine_spearman`) is returned.

## Updated `evaluate()` Dispatch (v3)

Now routes eight task types: `"Retrieval"`, `"Reranking"` (both to `evaluate_one`), `"PairClassification"`, `"MultilabelClassification"`, `"Clustering"`, `"Classification"`, `"STS"`, and `"Summarization"`. Unsupported types are skipped with a warning.

## Additional MTEB Reuse (Summarization)

- `AbsTaskSummarization.text_column_name` / `human_summaries_column_name` / `machine_summaries_column_name` / `relevancy_column_name` / `min_score` / `max_score` (task attributes for column names and score normalization)
- `SummarizationEvaluator` imported from `mteb._evaluators.text.summarization_evaluator` (available for reference; scoring logic is inlined to stay compatible with the DDP encoding pipeline)
- `scipy.stats.pearsonr` / `spearmanr` (correlation computation, matching `SummarizationEvaluator._calculate_metrics`)

---

## Sequence Length Filtering (8192 tokens)

Added filtering of all sequences exceeding 8192 tokens across every task type. The tokenization-aware filtering logic from `create_datasets.py` (`_remove_long_sequences`) was already in use for Retrieval; the change extends it uniformly and ensures label/pair/index consistency is maintained after removal.

### Core changes

- **`_prepare_text_dataset`** — `max_length` changed from `100_000` (effectively no filtering) to `8192`. Now returns a tuple `(dataset, removed_indices)` where `removed_indices` is a `set[int]` of original positional indices that were removed (too long or empty). All callers updated to unpack the tuple.
- **`_build_index_remap(n_original, removed_set)`** — new static helper that builds an `old_idx → new_idx` mapping after items are removed from a deduplicated text list. Used by PairClassification, STS, and Summarization to remap their index arrays.
- **`_prepare_retrieval`** — `max_length` changed from `4096` to `8192` for both queries and corpus. After filtering, removed query/corpus IDs are purged from `relevant_docs` (the qrels dict) so retrieval metrics are not penalized by queries that reference filtered documents. Queries left with no relevant docs after corpus filtering are also dropped.

### Per-task pairing maintenance

- **Classification / MultilabelClassification** — `train_labels` and `test_labels` are filtered in lockstep with their corresponding text datasets using the returned `removed_indices`.
- **Clustering** — `labels` list is filtered to match the surviving texts.
- **PairClassification** — after deduplication, pairs where either text was filtered are dropped. Remaining indices are remapped via `_build_index_remap` so they point to correct positions in the filtered dataset. Labels are filtered with the same pair mask.
- **STS** — same dedup-aware logic as PairClassification: invalid pairs are dropped, indices remapped, and `normalized_scores` filtered.
- **Summarization** — the most complex case. Per-sample human/machine summary index lists are walked: individual summaries pointing to removed texts are dropped, and samples that lose all human or all machine summaries are dropped entirely. `human_lens`, `machine_lens`, and `gold_scores` are rebuilt accordingly, and all surviving indices are remapped.
