# Refactor Summary Report

## Overview

This report documents the performance and correctness improvements made across three files in the F2LLM codebase. Changes focus on vectorization, memory efficiency, and cleaner tensor handling.

---

## `f2llm_repro/model.py` — 2 changes

### 1. `F2LLM.forward` — Vectorized last-token extraction

The old implementation used a Python `for`-loop that created **432 intermediate tensors** (one per sequence) before stacking them. This was replaced with a single vectorized index operation:

```python
# New
hidden[seq_range, last_idx]
```

The old code is preserved as comments above the new function.

> **Output shape change:** `[bs, 1, d]` → `[bs, d]`

---

### 2. `F2LLMT5Gemma2.forward` — Removed redundant unsqueeze

The `.unsqueeze(1)` calls on `query` and `passage` features were removed, as the shape change above makes them unnecessary. The old lines are preserved as inline comments.

> **Output shape change:** `[bs, 1, d]` → `[bs, d]`

---

## `train_f2llm_repro.py` — 1 change

### `_stack` — O(n²) → O(n) list flattening

The old implementation used `sum(data, [])`, which performs repeated list concatenation and scales as **O(n²)**. It was replaced with:

```python
# New
itertools.chain.from_iterable(data)  # O(n) single pass
```

The old version is preserved as a comment above the new implementation.

---

## `f2llm_repro/f2llm_train.py` — 2 changes

### 1. Training loop — Removed `.squeeze(1)` calls

Since the model now returns `[bs, d]` tensors directly (see model.py changes), all `.squeeze(1)` calls in the training loop were removed. Outputs are now unpacked into named variables — `q_feat`, `p_feat`, `n_feat` — to avoid repeating dictionary lookups.

### 2. `optimizer.zero_grad(set_to_none=True)`

Gradients are now set to `None` instead of being zeroed with `memset`. This avoids an unnecessary memory write on every backward pass, which is a standard PyTorch performance best practice.
