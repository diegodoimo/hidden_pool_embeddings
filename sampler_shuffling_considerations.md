# Dataset-Aware Batch Sampling

## Summary of Changes

### New Class: `DatasetAwareSampler` (`utils/dataloader_helpers.py`)

A DDP-aware sampler that guarantees each batch contains items from a **single dataset**. It supports two strategies:

**`sequential`**
All batches from dataset A are yielded before moving to dataset B, then C, etc. With `shuffle=True`, the order of datasets is permuted each epoch (via `set_epoch`). Within each dataset, items maintain their length-sorted order for optimal padding efficiency.

**`grouped`**
Batches from different datasets are interleaved in round-robin order, but each individual batch still contains items from only one dataset. This ensures the model sees diverse data throughout the epoch rather than focusing on one dataset at a time.

Both strategies guarantee:
- Every datapoint is processed **exactly once per epoch**
- Each dataset is padded to be divisible by `batch_size * num_replicas` so every batch is complete and homogeneous after DDP sharding
- Indices are interleaved across ranks (not chunked) to preserve balanced length distributions

---

### New Argument: `--batch_strategy` (`utils/arguments.py`)

| Value | Behavior |
|---|---|
| `mixed` *(default)* | Preserves existing `DistributedSampler` behavior |
| `sequential` | All batches from one dataset before the next |
| `grouped` | Round-robin interleaving across datasets |

---

### `train.py` Changes

- Imports `DatasetAwareSampler`
- Selects the appropriate sampler based on `--batch_strategy`
- Calls `set_epoch(epoch)` in the training loop (works for both `DistributedSampler` and `DatasetAwareSampler`)

---

### `utils/create_datasets.py` Change

Defensive addition: if a parquet file doesn't already have a `dataset_name` column, it is explicitly added from the directory path before concatenation. This ensures the sampler can always group by dataset.

---

## Design Decisions

### How Does `shuffle=True` Work?

Shuffling applies at the **dataset level**, not within individual datasets:

- **Sequential strategy:** Shuffles the *order of datasets* (e.g., D2 → D1 → D3), but items within each dataset keep their original length-sorted order.
- **Grouped strategy:** Shuffles the round-robin *starting order* of datasets, but items within each dataset stay in their length-sorted order.

There is no within-dataset shuffling in either case.

### Why Keep Length-Sorted Order?

Length sorting is the better default for this setup for three reasons:

1. **Padding efficiency** — The collate function pads all sequences in a batch to the length of the longest one. Mixing a 512-token document with several 50-token ones wastes ~90% of compute on padding. Length-sorted batches minimize this waste.

2. **Contrastive learning multiplies the cost** — Each example has a query + positive + N hard negatives, all padded independently. Padding waste is multiplied by `(2 + num_hard_negatives)` per example.

3. **Single-epoch training** — The main benefit of shuffling is breaking correlation across epochs. With `num_train_epochs=1` (the default), there are no repeated epochs to shuffle across.

#### When Would Within-Dataset Shuffling Help?

- **Multi-epoch training** where the model repeatedly sees the same length-correlated batches and could memorize order-dependent patterns
- If the loss curve shows sharp transitions at dataset boundaries (sequential strategy), shuffling within datasets would smooth it out

> **Bottom line:** For contrastive embedding training with variable-length text, the compute savings from length sorting typically outweigh the regularization benefit of shuffling.