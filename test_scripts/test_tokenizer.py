#!/usr/bin/env python3
"""Compare tokenized output in --output_dir against the tokenization produced
by main_test from f2llm_repro/tokenize_data_qwen.py.

Workflow
--------
1. Call main_test(args) on --root_dir (raw F2LLM parquet files).
   Returns {source_name: df} with columns query_input_ids, passage_input_ids,
   negative_1_input_ids, …, negative_N_input_ids.
2. For each source, load the corresponding saved parquet from --output_dir
   (produced by tokenize_data_f2llm.py --f2llm_prompt).
3. Compare token-ID columns row by row.

Usage (see also submit_tokenize_f2llm_test)
-------------------------------------------
    python test_tokenizer.py \\
        --tokenizer_path Qwen/Qwen3-0.6B \\
        --output_dir ./results/f2llm_tokenized/f2llm-prompt_qwen3-tok \\
        --root_dir /path/to/F2LLM/snapshots/<hash> \\
        [--datasets amazon_counterfactual fiqa] \\
        [--n_sample 500]
"""

import argparse
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f2llm_repro.tokenize_data_qwen import main_test


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_TOKENIZER   = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR  = "./results/f2llm_tokenized/f2llm-prompt_qwen3-tok"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_NUM_WORKERS = 1


# --------------------------------------------------------------------------- #
# Per-dataset comparison
# --------------------------------------------------------------------------- #
def check_dataset(source, df, output_dir, n_sample=None):
    out_path = os.path.join(output_dir, f"{source}.parquet")
    if not os.path.exists(out_path):
        print(f"  SKIP: saved parquet not found at {out_path}")
        return None

    tbl     = pq.read_table(out_path)
    saved_q = tbl["query_token_ids"].to_pylist()
    saved_p = tbl["positive_token_ids"].to_pylist()
    saved_n = tbl["negative_token_ids"].to_pylist()
    total_saved = len(saved_q)
    total_ref   = len(df)

    if total_ref != total_saved:
        print(
            f"  WARNING: row count mismatch — "
            f"root_dir parquet: {total_ref}, saved parquet: {total_saved}. "
            f"Comparing first {min(total_ref, total_saved)} rows."
        )

    num_neg = sum(1 for c in df.columns if c.startswith("negative_") and c.endswith("_input_ids"))
    n = min(n_sample, total_ref, total_saved) if n_sample is not None else min(total_ref, total_saved)
    print(f"  rows to check: {n} | negs/row: {num_neg}")

    errors = 0
    first_q_err = first_p_err = first_n_err = None

    for i in range(n):
        ref_q = [int(x) for x in df["query_input_ids"].iloc[i]]
        sav_q = [int(x) for x in saved_q[i]]
        if ref_q != sav_q:
            errors += 1
            if first_q_err is None:
                first_q_err = (i, ref_q, sav_q)

        ref_p = [int(x) for x in df["passage_input_ids"].iloc[i]]
        sav_p = [int(x) for x in saved_p[i]]
        if ref_p != sav_p:
            errors += 1
            if first_p_err is None:
                first_p_err = (i, ref_p, sav_p)

    for i in range(n):
        for j in range(1, num_neg + 1):
            ref_nj = [int(x) for x in df[f"negative_{j}_input_ids"].iloc[i]]
            sav_nj = [int(x) for x in saved_n[i][j - 1]]
            if ref_nj != sav_nj:
                errors += 1
                if first_n_err is None:
                    first_n_err = (i, j, ref_nj, sav_nj)

    if errors == 0:
        print(f"  PASS — all {n} rows match")
        return True

    print(f"  FAIL — {errors} mismatch(es) found")
    if first_q_err:
        i, r, s = first_q_err
        print(f"    first QUERY mismatch at row {i}:")
        print(f"      qwen-ref : {r[:8]}…")
        print(f"      saved    : {s[:8]}…")
    if first_p_err:
        i, r, s = first_p_err
        print(f"    first POSITIVE mismatch at row {i}:")
        print(f"      qwen-ref : {r[:8]}…")
        print(f"      saved    : {s[:8]}…")
    if first_n_err:
        i, j, r, s = first_n_err
        print(f"    first NEGATIVE mismatch at row {i}, neg {j}:")
        print(f"      qwen-ref : {r[:8]}…")
        print(f"      saved    : {s[:8]}…")
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root_dir",        required=True,
                        help="Directory with raw F2LLM parquet files (same as tokenize_data_qwen.py).")
    parser.add_argument("--output_dir",      default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory with tokenized parquets from tokenize_data_f2llm.py (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--tokenizer_path",  default=DEFAULT_TOKENIZER,
                        help=f"HuggingFace tokenizer path (default: {DEFAULT_TOKENIZER}).")
    parser.add_argument("--max_seq_len",     type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LEN}).")
    parser.add_argument("--num_workers",     type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"Parallel workers for tokenization (default: {DEFAULT_NUM_WORKERS}).")
    parser.add_argument("--datasets",        nargs="*", default=None,
                        help="Source names to test. Default: all parquets present in both root_dir and output_dir.")
    parser.add_argument("--n_sample",        type=int, default=None,
                        help="Max rows to check per dataset (default: all).")
    # accepted for compatibility with submit_tokenize_f2llm_test, not used by the test
    parser.add_argument("--f2llm_prompt",    action="store_true")
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    print(f"root_dir   : {args.root_dir}")
    print(f"output_dir : {args.output_dir}")
    print(f"tokenizer  : {args.tokenizer_path}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"num_workers: {args.num_workers}")
    print(f"n_sample   : {args.n_sample or 'all'}")

    # ---- 1. Run main_test on the full root_dir --------------------------------
    # Filter root_dir to only the requested datasets if --datasets is given
    if args.datasets:
        missing = [s for s in args.datasets if not os.path.exists(os.path.join(args.root_dir, f"{s}.parquet"))]
        if missing:
            print(f"WARNING: these datasets not found in root_dir: {missing}")

    all_dfs = main_test(args)   # {source_name: df}

    # ---- 2. Restrict to requested / available datasets -----------------------
    if args.datasets:
        sources = [s for s in args.datasets if s in all_dfs]
    else:
        sources = sorted(all_dfs.keys())

    # ---- 3. Compare each dataset against the saved parquet -------------------
    results = {}
    for src in sources:
        print(f"\n[{src}]")
        results[src] = check_dataset(src, all_dfs[src], args.output_dir, n_sample=args.n_sample)

    passed  = sum(1 for v in results.values() if v is True)
    failed  = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n{'=' * 50}")
    print(f"SUMMARY  passed={passed}  failed={failed}  skipped={skipped}")
    if failed:
        print("Failed datasets:")
        for k, v in results.items():
            if v is False:
                print(f"  - {k}")
        sys.exit(1)


if __name__ == "__main__":
    main()
