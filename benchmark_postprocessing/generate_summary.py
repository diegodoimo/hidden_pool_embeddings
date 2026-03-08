import json
import os

base = "results/datasets_negatives/qwen3_600m"

datasets = []
for root, dirs, files in os.walk(base):
    if "dataset_metadata.json" in files:
        path = os.path.join(root, "dataset_metadata.json")
        rel = os.path.relpath(root, base)
        parts = rel.split("/")

        with open(path) as f:
            meta = json.load(f)

        top_cat = parts[0]
        sub_cat = parts[1] if len(parts) > 1 else ""
        ds_name = parts[-1]

        datasets.append({
            "top_cat": top_cat,
            "sub_cat": sub_cat,
            "name": ds_name,
            "num_triples": meta["num_triples"],
            "num_with_15": meta["num_with_15_hard_negatives"],
        })

cat_map = {}
for d in datasets:
    if d["top_cat"] == "retrieval" and d["sub_cat"] == "summarization":
        task_type = "Summarization"
    elif d["top_cat"] == "retrieval":
        task_type = "Retrieval"
    elif d["top_cat"] == "sts":
        task_type = "STS"
    elif d["top_cat"] == "nli":
        task_type = "NLI"
    else:
        task_type = d["top_cat"]
    d["task_type"] = task_type

    if task_type not in cat_map:
        cat_map[task_type] = []
    cat_map[task_type].append(d)

for v in cat_map.values():
    v.sort(key=lambda x: x["num_with_15"], reverse=True)

all_sorted = sorted(datasets, key=lambda x: x["num_with_15"], reverse=True)

mteb_cats = ["Retrieval", "Clustering", "Reranking", "STS", "Classification", "PairClassification", "Summarization"]

lines = []
lines.append("=" * 110)
lines.append("QWEN3_600M DATASET SUMMARY: entries with 15 hard negatives")
lines.append("=" * 110)
lines.append("")

task_order = ["Retrieval", "Summarization", "STS", "NLI"]

for task_type in task_order:
    ds_list = cat_map.get(task_type, [])
    if not ds_list:
        continue

    total_triples = sum(d["num_triples"] for d in ds_list)
    total_15 = sum(d["num_with_15"] for d in ds_list)

    lines.append("-" * 110)
    lines.append(f"  {task_type}")
    lines.append("-" * 110)
    lines.append(f"  {'sub_category':<30s} {'dataset':<30s} {'num_triples':>15s} {'num_with_15_neg':>18s}")
    lines.append(f"  {'-'*30} {'-'*30} {'-'*15} {'-'*18}")

    for d in ds_list:
        sub = d["sub_cat"] if d["sub_cat"] else "-"
        lines.append(f"  {sub:<30s} {d['name']:<30s} {d['num_triples']:>15,d} {d['num_with_15']:>18,d}")

    lines.append(f"  {'':<30s} {'TOTAL ' + task_type:<30s} {total_triples:>15,d} {total_15:>18,d}")
    lines.append("")

for cat in mteb_cats:
    if cat not in cat_map:
        lines.append(f"  {cat}: no training datasets")
lines.append("")

lines.append("=" * 110)
lines.append("ALL DATASETS SORTED BY DECREASING num_with_15_hard_negatives")
lines.append("=" * 110)
lines.append(f"  {'task_type':<15s} {'sub_category':<30s} {'dataset':<30s} {'num_triples':>15s} {'num_with_15_neg':>18s}")
lines.append(f"  {'-'*15} {'-'*30} {'-'*30} {'-'*15} {'-'*18}")

for d in all_sorted:
    sub = d["sub_cat"] if d["sub_cat"] else "-"
    lines.append(f"  {d['task_type']:<15s} {sub:<30s} {d['name']:<30s} {d['num_triples']:>15,d} {d['num_with_15']:>18,d}")

lines.append("")

lines.append("=" * 110)
lines.append("TOTALS BY TASK TYPE")
lines.append("=" * 110)
lines.append(f"  {'task_type':<25s} {'num_triples':>15s} {'num_with_15_neg':>18s}")
lines.append(f"  {'-'*25} {'-'*15} {'-'*18}")

grand_triples = 0
grand_15 = 0
type_totals = []
for task_type in task_order:
    ds_list = cat_map.get(task_type, [])
    t = sum(d["num_triples"] for d in ds_list)
    n = sum(d["num_with_15"] for d in ds_list)
    type_totals.append((task_type, t, n))
    grand_triples += t
    grand_15 += n

type_totals.sort(key=lambda x: x[2], reverse=True)
for task_type, t, n in type_totals:
    lines.append(f"  {task_type:<25s} {t:>15,d} {n:>18,d}")

lines.append(f"  {'-'*25} {'-'*15} {'-'*18}")
lines.append(f"  {'GRAND TOTAL':<25s} {grand_triples:>15,d} {grand_15:>18,d}")
lines.append("")

output = "\n".join(lines)
print(output)

with open("qwen3_600m_dataset_negatives_summary.txt", "w") as f:
    f.write(output + "\n")

print("\nWritten to qwen3_600m_dataset_negatives_summary.txt")
