"""
Download/cache the F2LLM dataset from HuggingFace.

F2LLM has parquet files with heterogeneous schemas (3, 26, 27, or 28 columns),
so load_dataset() fails with CastError when unifying them. We load each
parquet separately and concatenate with a normalized schema.
"""
import os
from datasets import load_dataset, Dataset, concatenate_datasets, Value, Features

# Target schema: superset of all variants (query, passage, negative_1..24)
TARGET_FEATURES = Features({
    "query": Value("string"),
    "passage": Value("string"),
    **{f"negative_{i}": Value("string") for i in range(1, 25)},
})


def _normalize_to_target(ds: Dataset) -> Dataset:
    """Map dataset columns to target schema; add empty strings for missing negatives."""
    cols = set(ds.column_names)
    keep = ["query", "passage"] + [f"negative_{i}" for i in range(1, 25) if f"negative_{i}" in cols]
    out = ds.select_columns(keep)
    for i in range(1, 25):
        if f"negative_{i}" not in cols:
            out = out.add_column(f"negative_{i}", [""] * len(out))
    return out.cast(TARGET_FEATURES)


def _get_f2llm_snapshot_path():
    """Return path to F2LLM snapshot directory."""
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base = os.path.join(
        cache_dir,
        "hub",
        "datasets--codefuse-ai--F2LLM",
        "snapshots",
    )
    if not os.path.isdir(base):
        try:
            load_dataset("codefuse-ai/F2LLM")
        except Exception:
            pass
        base = os.path.join(cache_dir, "hub", "datasets--codefuse-ai--F2LLM", "snapshots")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No F2LLM cache at {base}; run with HF access to download")
    snapshot_dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not snapshot_dirs:
        raise FileNotFoundError(f"No F2LLM snapshot in {base}")
    return os.path.join(base, snapshot_dirs[0])


def load_f2llm(return_per_source=False, sources=None):
    """
    Load F2LLM, handling schema mismatches across parquet files.

    Args:
        return_per_source: If True, also return list of (parquet_name, count) per source.
        sources: Optional list of source names to load (e.g. ['amazon_qa', 'arguana']).
                 If None, load all sources.

    Returns:
        Dataset, or (Dataset, per_source_list) if return_per_source=True.
    """
    snapshot = _get_f2llm_snapshot_path()
    all_parquets = sorted([f for f in os.listdir(snapshot) if f.endswith(".parquet")])
    if not all_parquets:
        raise FileNotFoundError(f"No parquet files in {snapshot}")

    if sources is not None:
        want = set(s.strip() for s in sources)
        parquets = [f for f in all_parquets if f.replace(".parquet", "") in want]
        missing = want - {f.replace(".parquet", "") for f in parquets}
        if missing:
            raise ValueError(f"Unknown or missing sources: {sorted(missing)}")
    else:
        parquets = all_parquets

    parts = []
    per_source = []
    for pf in parquets:
        path = os.path.join(snapshot, pf)
        name = pf.replace(".parquet", "")
        try:
            ds = Dataset.from_parquet(path)
        except Exception:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            table = table.replace_schema_metadata({})
            ds = Dataset(table)
        per_source.append((name, len(ds)))
        parts.append(_normalize_to_target(ds))

    data = concatenate_datasets(parts)
    if return_per_source:
        return data, per_source
    return data


def inspect_f2llm_metadata(sources=None):
    """
    Inspect F2LLM structure without loading any row data.
    Uses parquet file metadata only (schema + row counts in footer).

    Args:
        sources: Optional list of source names to inspect. If None, inspect all.
    """
    import pyarrow.parquet as pq

    snapshot = _get_f2llm_snapshot_path()
    all_parquets = sorted([f for f in os.listdir(snapshot) if f.endswith(".parquet")])
    if sources is not None:
        want = set(s.strip() for s in sources)
        parquets = [f for f in all_parquets if f.replace(".parquet", "") in want]
        missing = want - {f.replace(".parquet", "") for f in parquets}
        if missing:
            raise ValueError(f"Unknown or missing sources: {sorted(missing)}")
    else:
        parquets = all_parquets

    if not parquets:
        raise FileNotFoundError(f"No parquet files in {snapshot}")

    per_source = []
    schemas = {}
    for pf in parquets:
        path = os.path.join(snapshot, pf)
        name = pf.replace(".parquet", "")
        pf_obj = pq.ParquetFile(path)
        nrows = pf_obj.metadata.num_rows
        per_source.append((name, nrows))
        schemas[name] = pf_obj.schema.names

    total = sum(c for _, c in per_source)
    print("=" * 60)
    print("F2LLM METADATA (no data loaded)")
    print("=" * 60)
    print(f"\nTotal examples: {total:,}")
    print(f"Number of parquet sources: {len(parquets)}")
    print("\nPer-source breakdown (rows):")
    for name, count in per_source:
        print(f"  - {name}: {count:,} examples")
    print("\nSchema variants (columns per source):")
    by_cols = {}
    for name, cols in schemas.items():
        key = tuple(cols)
        if key not in by_cols:
            by_cols[key] = []
        by_cols[key].append(name)
    for cols, names in sorted(by_cols.items(), key=lambda x: -len(x[0])):
        print(f"  {len(cols)} cols: {cols[:5]}{'...' if len(cols) > 5 else ''}")
        print(f"    Sources: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
    print("=" * 60)


def inspect_dataset(data, per_source=None):
    """
    Print dataset structure: fields, splits, and number of datapoints per split.

    Args:
        data: HuggingFace Dataset or DatasetDict
        per_source: Optional list of (name, count) for per-source breakdown (e.g. parquet files)
    """
    from datasets import DatasetDict

    print("=" * 60)
    print("DATASET STRUCTURE")
    print("=" * 60)

    if isinstance(data, DatasetDict):
        print("\nSplits:")
        for split_name, ds in data.items():
            print(f"  - {split_name}: {len(ds):,} examples")
        print("\nFields (from first split):")
        first = next(iter(data.values()))
        for name, dtype in first.features.items():
            print(f"  - {name}: {dtype}")
    else:
        print(f"\nTotal examples: {len(data):,}")
        print("\nFields:")
        for name, dtype in data.features.items():
            print(f"  - {name}: {dtype}")

    if per_source:
        print("\nPer-source breakdown:")
        for name, count in per_source:
            print(f"  - {name}: {count:,} examples")
        print(f"  TOTAL: {sum(c for _, c in per_source):,}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download/inspect F2LLM dataset")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Full load: load all data into memory (default: metadata-only inspection)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Load only these sources (e.g. amazon_qa arguana msmarco)",
    )
    args = parser.parse_args()
    if args.load:
        data, per_source = load_f2llm(return_per_source=True, sources=args.sources)
        inspect_dataset(data, per_source=per_source)
    else:
        inspect_f2llm_metadata(sources=args.sources)