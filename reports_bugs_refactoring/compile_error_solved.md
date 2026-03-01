# Bug Report: Inplace Operations Breaking `torch.compile` Backward Pass

## Summary

Inplace tensor operations in `pairwise_dot_squared` methods cause a `RuntimeError` during the backward pass when combined with `torch.compile`. Autograd detects that tensors saved for backward have been mutated in-place, breaking gradient computation.

---

## Root Cause

Both `EmbeddingGemmaLossDistributed` and `EmbeddingGemmaLossHardNegatives` use inplace operations (`fill_diagonal_()`, `masked_fill_()`) on `dots_sq` — a tensor that lives in the autograd computation graph (derived from model outputs via `x @ x.t()` then `**2`).

When `torch.compile` traces the backward pass, it records tensor versions and detects the mutation, resulting in a `RuntimeError`.

---

## Affected Code — `utils/losses.py`

### `EmbeddingGemmaLossDistributed.pairwise_dot_squared` (line 50)

```python
# BEFORE (broken)
dots_sq.fill_diagonal_(0)
return dots_sq.sum() / (B * (B - 1))
```

### `EmbeddingGemmaLossHardNegatives.pairwise_dot_squared` (lines 125, 129)

```python
# BEFORE (broken)
dots_sq = dots**2
dots_sq.fill_diagonal_(0)
same_id = ids.unsqueeze(1) == ids.unsqueeze(0)
same_id.fill_diagonal_(False)
dots_sq.masked_fill_(same_id, 0)
```

---

## Fix

Replace all inplace operations on gradient-tracked tensors with their out-of-place equivalents.

### `EmbeddingGemmaLossDistributed` (+2 / -2)

```python
# AFTER (fixed)
dots_sq = dots**2
diag_mask = torch.eye(B, dtype=torch.bool, device=dots_sq.device)
dots_sq = dots_sq.masked_fill(diag_mask, 0)
return dots_sq.sum() / (B * (B - 1))
```

### `EmbeddingGemmaLossHardNegatives` (+4 / -5)

```python
# AFTER (fixed)
dots_sq = dots**2
diag_mask = torch.eye(B, dtype=torch.bool, device=dots_sq.device)
same_id = ids.unsqueeze(1) == ids.unsqueeze(0)
same_id = same_id.masked_fill(diag_mask, False)
exclude = diag_mask | same_id
dots_sq = dots_sq.masked_fill(exclude, 0)
```

---

## Changes at a Glance

| Inplace (broken) | Out-of-place (fixed) |
|---|---|
| `dots_sq.fill_diagonal_(0)` | `dots_sq = dots_sq.masked_fill(diag_mask, 0)` using `torch.eye` boolean mask |
| `dots_sq.masked_fill_(same_id, 0)` | Merged into single `dots_sq.masked_fill(exclude, 0)` call |
| `same_id.fill_diagonal_(False)` | `same_id = same_id.masked_fill(diag_mask, False)` |

---

## Additional Fix — `train.py`: Dual Model Calls

### Root Cause

The `torch.compile`-wrapped model was being called twice within the same backward graph — once for queries, once for documents. AOTAutograd saves tensors during the first forward call for use in the backward pass. When the second forward call goes through the same compiled function, shared module state (such as the scalar `embed_scale` buffer) can have its version counter incremented, invalidating the first call's saved tensors.

### Fix

Instead of two separate model calls, query and document inputs are now concatenated into a single batch, passed through the model once, then split:

```python
# BEFORE (two calls → version mismatch)
query_embeddings = self.model(query_inputs, query_mask)
doc_embeddings = self.model(doc_inputs, doc_mask)

# AFTER (single call → no conflict)
combined_out = self.model(
    torch.cat([query_inputs, doc_inputs]),
    torch.cat([query_mask, doc_mask]),
)
query_embeddings = combined_out[:B]
doc_embeddings = combined_out[B:]
```

This is the standard pattern for contrastive learning with `torch.compile`. Both the hard-negatives and non-hard-negatives code paths have been updated.

---

## Combined Effect

Together, these two fixes fully resolve the inplace version mismatch errors:

| Fix | File | Issue |
|---|---|---|
| Out-of-place tensor ops | `utils/losses.py` | `fill_diagonal_` / `masked_fill_` mutating tensors in the autograd graph |
| Single batched model call | `train.py` | Dual forward passes incrementing shared buffer version counters |

---

## Scope Note

The remaining `fill_diagonal_(False)` calls in forward methods (on `doc_id_matches`, `dup_doc`, `dup_query`, `similarity_mask`) are **safe** — those tensors are produced by comparison operations and do not require gradients.