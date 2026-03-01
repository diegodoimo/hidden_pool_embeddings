# Training with Hard Negatives — Implementation Report

## Overview

Added support for training on hard-negative datasets stored as parquet files
under `datasets_negatives/<model_name>/`. Each dataset contains query-positive-negative
triples with up to 24 mined hard negatives per example. The first N (default 8)
hard negatives are used per training example.

---

## Dataset format

Each parquet file (`data.parquet`) has columns:

| Column | Type | Required |
|---|---|---|
| `query_text` | `string` | yes |
| `query_id` | `string` | yes |
| `positive_text` | `string` | yes |
| `positive_id` | `string` | yes |
| `negative_text` | `list<string>` | yes |
| `negative_id` | `list<string>` | yes |
| `positive_title` | `string` | optional |
| `negative_title` | `list<string>` | optional |

Directory layout under `datasets_negatives/qwen3_600m/`:

```
retrieval/
  general_retrieval/    arguana, msmarco, nfcorpus, stackexchange, mrtydi
  domain_specific_qa/   amazonqa, fiqa2018, pubmedqa
  open_domain_qa/       naturalquestions, squad, eli5, triviaqa, hotpotqa
  fact_verification/    scifact, fever
  paraphrase_detection/ qqp, stackexchange_dup_s2s/p2p, stackoverflow_dup
  scientific_doc_retrieval/ specter
  summarization/        xsum, cnndm, sentence_compression, wikihow
nli/                    snli, mnli, anli
sts/                    sts12, stsbenchmark, sts22
```

---

## Files modified

### 1. `utils/arguments.py`

New CLI arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--negatives_dir` | `str` | `None` | Root directory of hard-negative parquet datasets |
| `--num_hard_negatives` | `int` | `8` | Number of hard negatives per training example |
| `--max_query_len` | `int` | `256` | Max token length for queries |
| `--max_passage_len` | `int` | `512` | Max token length for positives/negatives |
| `--instruction_template` | `str` | `qwen3` | Prompt template (`qwen3` or `embeddinggemma`) |

### 2. `utils/contrastive_datasets.py`

New components added at the end of the file:

- **`TrainTaskMetadata`** — lightweight dataclass with `.type` and `.prompt` fields,
  mimicking the MTEB metadata interface so the same `instruction_template_qwen3` /
  `instruction_template_embeddinggemma` functions from `create_datasets.py` can be reused.

- **`FOLDER_TO_TASK`** — mapping from directory path prefixes to `TrainTaskMetadata`.
  Task-specific prompts are assigned per subfolder:
  - `retrieval/general_retrieval` → generic retrieval instruction
  - `retrieval/domain_specific_qa`, `retrieval/open_domain_qa` → QA instruction
  - `retrieval/fact_verification` → claim verification instruction
  - `retrieval/paraphrase_detection` → semantic similarity instruction
  - `retrieval/summarization` → summarization instruction
  - `nli` → NLI entailment instruction
  - `sts` → semantic similarity instruction

- **`_infer_task_metadata(parquet_path, base_dir)`** — walks the relative path from
  deepest to shallowest folder to find the best matching task metadata.

- **`_load_parquet_safe(path)`** — tries `Dataset.from_parquet()`, falls back to
  stripping HF metadata via pyarrow if the datasets library version is incompatible.

- **`_str_to_int_id(s)`** — deterministic MD5 hash to convert string document IDs
  (e.g. `"doc_0"`) into 63-bit integers for the loss function's duplicate masking.
  IDs are namespaced by dataset name to avoid cross-dataset collisions.

- **`tokenize_hard_negatives_batch()`** — core tokenization function (used inside
  `Dataset.map(batched=True)`). For each example:
  1. Builds query prompt via `instruction_template(PromptType.query, ...)`
  2. Builds positive prompt via `instruction_template(PromptType.document, ..., title=...)`
  3. Builds negative prompts (first `num_hard_negatives`) the same way
  4. Tokenizes with truncation to `max_query_len` / `max_passage_len`
  5. If fewer negatives than required, pads with the positive's tokens
  6. Outputs: `query_token_ids`, `pos_token_ids`, `neg_token_ids` (list of lists),
     `pos_ids`, `query_len`, `pos_len`, `total_len`

- **`load_hard_negatives_datasets()`** — orchestrator function:
  1. Recursively finds all `data.parquet` files under `base_dir`
  2. For each file: infers task metadata, loads the parquet, filters examples with
     fewer than `num_hard_negatives` negatives, tokenizes
  3. Concatenates all tokenized datasets via `concatenate_datasets()`
  4. Sorts by `total_len` (descending) for length-balanced batching
  5. Prints summary statistics (total examples, tokens, avg lengths)

- **`collate_fn_with_hard_negatives()`** — collate function for the DataLoader:
  - Pads queries, positives, and negatives independently with `pad_sequence`
  - Negatives are flattened across the batch `(B * num_neg)`, padded to the longest,
    then reshaped to `(B, num_hard_negatives, seq_len)`
  - Returns attention masks for all three groups

### 3. `train.py`

- **Imports**: added `collate_fn_with_hard_negatives`, `load_hard_negatives_datasets`,
  `functools.partial`.

- **Data loading** (`main()`):
  - Selects `instruction_template_qwen3` or `instruction_template_embeddinggemma`
    based on `args.instruction_template`
  - Calls `load_hard_negatives_datasets()` with the chosen template
  - Sets the collate function to `collate_fn_with_hard_negatives` via `partial`
    (passing `pad_token_id` and `num_hard_negatives`)

- **Loss initialization**:
  - `EmbeddingGemmaLossHardNegatives` is now initialized with
    `num_hard_negatives=args.num_hard_negatives` (was hardcoded to 7)

- **Training loop**:
  - If `neg_token_ids` is in the batch **and** the loss is `EmbeddingGemmaLossHardNegatives`:
    reshapes negatives `(B, N, L) → (B*N, L)`, forwards through the model,
    reshapes embeddings back to `(B, N, D)`, passes to loss as `hard_neg_embeddings`
  - Otherwise (e.g. `EmbeddingGemmaLossDistributed`): uses in-batch negatives only

- **Bug fix**: `args.only_eval` → `args.eval_only` (matching the argument definition)

---

## Usage

```bash
# Train on all qwen3_600m datasets with 8 hard negatives
torchrun --nproc_per_node=4 train.py \
    --model_name_or_path <model_path> \
    --negatives_dir results/datasets_negatives/qwen3_600m \
    --num_hard_negatives 8 \
    --instruction_template qwen3 \
    --max_query_len 256 \
    --max_passage_len 512 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --output_dir results/train_output \
    --use_lora

# Use ALL model directories at once
torchrun --nproc_per_node=4 train.py \
    --negatives_dir results/datasets_negatives \
    ...
```

## Notes

- Each training step now forwards `B + B + B*N` sequences through the model
  (queries + positives + negatives). With `N=8` this is 10x the original cost
  per step, so reduce `per_device_train_batch_size` accordingly.
- The `FOLDER_TO_TASK` mapping in `contrastive_datasets.py` can be extended for
  new task categories without touching any other code.
- String document IDs are hashed to integers and namespaced by dataset path, so
  duplicate masking in the loss works correctly across concatenated datasets.
