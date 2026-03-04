# GPU Utilization Optimization Report

## Problem Summary

The GPU is idle approximately 30% of the time due to CPU-side overhead between kernel launches — not a lack of compute. The bottleneck is not arithmetic intensity but the gaps between dispatched operations.

---

## Root Causes

**Two separate forward passes per step** — Python returns control between the query and doc forward calls, creating a bubble where the GPU stalls waiting for the next kernel dispatch.

**Default `torch.compile` mode** — Does not capture CUDA graphs. Each compiled op still requires a separate kernel launch from the CPU. With a 270M model and short sequences, kernel launch overhead dominates over actual compute time.

**Synchronous `.to(device)` transfers** — Blocks the CPU until the H2D copy finishes, preventing overlap with the previous step's backward/optimizer pass.

**Per-parameter optimizer kernels** — Standard AdamW launches one CUDA kernel per parameter tensor (hundreds of tensors), creating significant cumulative launch overhead.

**DDP recomputes bucket structure every step** — Unnecessary when the forward graph is fixed across steps.

---

## Optimizations Applied

### 1. Fused AdamW (`optimizer.py:57`)
Added `fused=True` to the AdamW optimizer. Instead of launching one CUDA kernel per parameter tensor, a single multi-tensor kernel handles all parameters at once. This is the biggest low-hanging fruit given the number of parameter tensors involved.

### 2. Fused Query + Doc Forward Pass (`train.py:282–296`)
Previously, two separate `self.model()` calls were made — one for 16 queries, one for 128 docs — with a Python-level gap between them. These are now fused into a single forward call, eliminating the inter-call bubble and allowing `torch.compile` to optimize one unified graph instead of two.

### 3. Non-Blocking H2D Transfers (`train.py:261`)
Added `non_blocking=True` to `.to(device)` calls. Combined with the already-enabled `pin_memory=True` in the DataLoader, this overlaps host-to-device memory copies with the previous step's compute, removing a synchronization stall.

### 4. `torch.compile` with `max-autotune` (`train.py:153`)
Switched from default compile mode to `mode="max-autotune"`. This enables CUDA graph capture and autotuned Triton kernels, reducing per-op kernel launch overhead across the entire forward/backward graph to near zero. This is the single highest-impact knob for GPU utilization.

> **Note:** The first few steps will be slower as `max-autotune` benchmarks multiple kernel configurations. This is a one-time warmup cost. For faster startup with most of the benefit, `mode="reduce-overhead"` captures CUDA graphs while skipping Triton autotuning.

### 5. DDP `static_graph=True` + `gradient_as_bucket_view=True` (`train.py:148–152`)
Since the forward graph never changes shape, `static_graph=True` tells DDP to cache the bucket reconstruction and allreduce schedule rather than recomputing it each step. `gradient_as_bucket_view=True` additionally avoids copying gradients into allreduce buckets.

---

## Summary Table

| # | File | Change | Expected Impact |
|---|------|--------|-----------------|
| 1 | `optimizer.py:57` | `fused=True` on AdamW | ~100s of per-param kernels → 1 multi-tensor kernel |
| 2 | `train.py:282–296` | Fused query+doc into single forward pass | Eliminates Python-level gap; one compiled graph |
| 3 | `train.py:261` | `non_blocking=True` on `.to(device)` | Overlaps H2D copy with previous step's compute |
| 4 | `train.py:153` | `mode="max-autotune"` on `torch.compile` | CUDA graph capture + autotuned kernels |
| 5 | `train.py:148–152` | `static_graph=True` + `gradient_as_bucket_view=True` | Caches allreduce schedule; avoids gradient copy |

---

## Expected Outcome

After warmup, GPU power draw should climb from ~500W to **650–700W** as kernel launch gaps are eliminated.

---

## `torch.compile` and Multiple Forward Passes

A natural question is whether a function like `_step` can be compiled when it contains two sequential `model()` calls. The answer is yes — `torch.compile` traces the entire Python function into a single FX graph, and two forward passes inside `_step` are treated as one continuous dataflow graph.

**Same parameters, one graph.** Since both calls go through the same DDP-wrapped model, the compiler knows the weights are shared. It traces both calls and their interaction with the loss into a single compiled graph and a single fused backward.

**No graph break.** Graph breaks occur from data-dependent Python control flow (e.g. `if tensor.item() > 0`), print statements, or unsupported ops. Two sequential calls to the same module are just more nodes in the same graph.

**Two calls ≠ two compilations.** The compiler traces `_step` once and produces one optimized kernel plan. The two `model()` invocations become two subgraphs within that single plan, and with `max-autotune` the whole thing can be captured into one CUDA graph.

**Backward is also unified.** Since the loss depends on outputs from both forward calls, `loss.backward()` produces a single backward graph through both forward passes and the loss, which the compiler optimizes as a whole.

The only scenario where this could cause a recompilation is if `num_neg` (a plain Python int from the batch dict) changes value between steps — but since `num_hard_negatives` is fixed, that won't happen.

---

## Further Consideration

With a batch size of 16 and 7 hard negatives, each step processes 144 sequences (128 docs + 16 queries) at `max_seq_len=1024` on a 270M model. Some SMs may still be underutilized at this scale. If memory allows, increasing `per_device_train_batch_size` to 32 or 64 would raise arithmetic intensity and push utilization higher.