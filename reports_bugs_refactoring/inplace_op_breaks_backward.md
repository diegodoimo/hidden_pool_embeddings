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

## Scope Note

The remaining `fill_diagonal_(False)` calls in forward methods (on `doc_id_matches`, `dup_doc`, `dup_query`, `similarity_mask`) are **safe** — those tensors are produced by comparison operations and do not require gradients.