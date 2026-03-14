import argparse
import glob
import json
import os
from typing import List, Dict, Any


def _discover_files(input_dir: str, run_label: str = "") -> List[str]:
    if run_label:
        pattern = os.path.join(input_dir, f"batch_sample_map_{run_label}_rank*.jsonl")
    else:
        pattern = os.path.join(input_dir, "batch_sample_map*_rank*.jsonl")
    return sorted(glob.glob(pattern))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_batch_maps(input_dir: str, run_label: str = "") -> List[Dict[str, Any]]:
    files = _discover_files(input_dir, run_label=run_label)
    if not files:
        raise FileNotFoundError(
            f"No batch sample map files found in '{input_dir}'"
            + (f" for run_label='{run_label}'" if run_label else "")
        )

    merged: List[Dict[str, Any]] = []
    for file_path in files:
        merged.extend(_read_jsonl(file_path))

    merged.sort(key=lambda x: (x.get("batch_id", -1), x.get("rank", -1)))
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-rank batch sample map JSONL files into one file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing batch_sample_map*_rank*.jsonl files.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="Optional run label used in filenames (matches --out_filename after suffix expansion).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path. Defaults to merged_batch_sample_map[_{run_label}].jsonl in input_dir.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json"],
        help="Output format.",
    )
    args = parser.parse_args()

    merged = merge_batch_maps(args.input_dir, run_label=args.run_label)

    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.run_label}" if args.run_label else ""
        ext = "jsonl" if args.format == "jsonl" else "json"
        output_path = os.path.join(
            args.input_dir, f"merged_batch_sample_map{suffix}.{ext}"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.format == "jsonl":
        with open(output_path, "w") as f:
            for row in merged:
                f.write(json.dumps(row) + "\n")
    else:
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)

    print(f"Merged {len(merged)} records to {output_path}")


if __name__ == "__main__":
    main()
