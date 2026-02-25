# Retrieval Performance Bug — Missing All-Gather in `search()`

## Symptom

All retrieval task scores are roughly halved compared to expected values when running with `world_size > 1` (e.g. 2 GPUs).

| Task | Expected | Observed (2 GPUs) |
|------|----------|--------------------|
| ArguAna | ~0.69 | 0.387 |
| FiQA2018 | ~0.50 | 0.229 |
| CQADupstackGamingRetrieval | — | 0.373 |
| FEVERHardNegatives | — | 0.474 |

Non-retrieval tasks (Classification, Clustering, STS, PairClassification, Summarization) are unaffected because they use `_encode_dataset` / `encode()` **without** `divided_by_chunks`, which correctly all-gathers.

## Root Cause

**File:** `inference/helpers.py`, function `search()`, lines ~265-278.

The corpus encoding inside the search loop calls `encode()` with `divided_by_chunks=True`:

```python
local_corpus_chunk, local_indices = encode(
    model,
    corpus_loader,
    prompt_type=PromptType.document,
    world_size=world_size,
    divided_by_chunks=True,   # <-- returns LOCAL embeddings only
    pool_fn=pool_fn,
)
```

When `divided_by_chunks=True`, `encode()` returns early **before** the all-gather:

```python
# in encode():
if prompt_type == PromptType.document and divided_by_chunks:
    return embeddings, indices   # LOCAL only, no all-gather
```

The `LenghtSortedSampler` distributes corpus items across ranks (each rank gets `1/world_size` of the subcorpus chunk). So `local_corpus_chunk` on each rank contains only its share.

Then:

```python
scores = torch.matmul(q_slice, local_corpus_chunk.T)
```

Each rank computes similarity scores against only **its own fraction** of the corpus chunk. There is no subsequent merge or all-gather of `top_scores` / `top_indices` across ranks. Each rank independently maintains an incomplete top-k that only covers `1/world_size` of the corpus.

With 2 GPUs, each rank only sees ~50% of the documents, which explains the ~50% performance drop.

**Query embeddings are fine** — they use `encode()` without `divided_by_chunks`, which correctly all-gathers so every rank has all query embeddings.

## Why Other Tasks Are Unaffected

All non-retrieval tasks use `_encode_dataset()` which calls `encode()` with the default `divided_by_chunks=False`. This triggers the all-gather inside `encode()`, so every rank ends up with the full embedding matrix. The retrieval-specific `search()` function is the only code path that uses `divided_by_chunks=True`.

## Fix

Add an all-gather of the corpus chunk **inside** the search loop, between the `encode()` call and the similarity computation. This preserves the DDP encoding speedup (each rank only runs the model forward pass on its share) while ensuring every rank has the complete corpus chunk for scoring.

In `inference/helpers.py`, after the `encode()` call inside `search()`:

```python
local_corpus_chunk, local_indices = encode(
    model,
    corpus_loader,
    prompt_type=PromptType.document,
    world_size=world_size,
    divided_by_chunks=True,
    pool_fn=pool_fn,
)

# --- ADD THIS BLOCK ---
# All-gather corpus chunk so every rank has the full chunk.
if world_size > 1:
    chunk_n = len(subcorpus)
    gathered_emb = [torch.zeros_like(local_corpus_chunk) for _ in range(world_size)]
    gathered_idx = [torch.zeros_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered_emb, local_corpus_chunk)
    dist.all_gather(gathered_idx, local_indices)
    local_corpus_chunk = torch.cat(gathered_emb, dim=0)[:chunk_n]
    local_indices = torch.cat(gathered_idx, dim=0)[:chunk_n]
    del gathered_emb, gathered_idx
# ----------------------

global_indices = local_indices + chunk_idx
```

### Why `[:chunk_n]` is needed

`LenghtSortedSampler` pads indices to make the total divisible by `num_replicas`. After all-gather, the concatenated tensor has `num_samples * world_size` rows, which may exceed the actual subcorpus size. Trimming with `[:len(subcorpus)]` removes the padded duplicates.

### Memory impact

Minimal — the all-gather doubles the corpus chunk on GPU momentarily, but `chunk_size` is typically 50k embeddings (~400 MB at float32 × 1536-dim), well within the ~80 GB H100 budget. The intermediate `gathered_emb` / `gathered_idx` lists are deleted immediately.
