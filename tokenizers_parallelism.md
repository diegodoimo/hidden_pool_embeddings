# Fix: HuggingFace Tokenizer Parallelism Warning

## Background

This is a well-known warning, not a bug. It occurs when HuggingFace tokenizers use
Rust-level parallelism and the process is subsequently forked — for example, by a
`DataLoader` with `num_workers > 0`. The fix is to set the `TOKENIZERS_PARALLELISM`
environment variable early in the script, before any forking takes place.

## The Fix

Add one line near the top of `train.py`, right after `import os`:

```python
import os

# Disable Rust-level tokenizer parallelism to avoid deadlocks when the process
# is forked by DataLoader workers or DDP (the tokenizer is used inside collate_fn).
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

Setting it to `"false"` is safe because PyTorch's `DataLoader` already handles
parallelism at the Python level — there is no performance cost.

## Where the Issue Occurs in `train.py`

Three locations in the script contribute to the problem:

- **Line 309** — `AutoTokenizer.from_pretrained(...)` loads the tokenizer, which
  initializes Rust-level parallel threads internally.
- **Line 295** — `dist.init_process_group(...)` forks or spawns processes for DDP.
- **Lines 411–416** — `DataLoader(..., num_workers=args.num_workers)` forks worker
  processes.

Because the tokenizer is captured inside `collate_fn` via `partial(...)` (line 406)
and used inside `DataLoader` workers, the Rust threads initialized in the parent
process conflict with the forked workers. That conflict is the source of the warning
and can cause deadlocks.

## Summary

Setting `TOKENIZERS_PARALLELISM=false` at the top of the script tells the tokenizer
never to spin up Rust threads, making it safe to fork. PyTorch's `DataLoader` workers
already provide all the parallelism needed, so nothing is lost.