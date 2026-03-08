import json


def print_summary(json_path: str, output_path: str | None = None):
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    all_scores = []          # for global micro average
    type_averages = []       # for global macro average

    for task_type, tasks in data.items():
        scores = []
        for task_dict in tasks:
            for task_name, (metric_dict, _) in task_dict.items():
                score = list(metric_dict.values())[0] * 100
                scores.append(score)
        type_avg = sum(scores) / len(scores)
        type_averages.append(type_avg)
        all_scores.extend(scores)

        lines.append(f"{'='*60}")
        lines.append(f"Task type: {task_type}  (n={len(scores)})")
        lines.append(f"{'='*60}")
        for task_dict in tasks:
            for task_name, (metric_dict, _) in task_dict.items():
                metric_name = list(metric_dict.keys())[0]
                score = list(metric_dict.values())[0] * 100
                lines.append(f"  {task_name:<50s}  {metric_name}: {score:.2f}")
        lines.append(f"  {'Average':<50s}  {type_avg:.2f}")
        lines.append("")

    global_micro = sum(all_scores) / len(all_scores)
    global_macro = sum(type_averages) / len(type_averages)

    lines.append("=" * 60)
    lines.append("GLOBAL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Micro average (all tasks equally weighted):  {global_micro:.2f}")
    lines.append(f"  Macro average (task-type averages averaged): {global_macro:.2f}")

    report = "\n".join(lines)
    print(report)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    print_summary(
        "qwen3_embedding_mteb_eng_v2_results.json",
        "qwen3_embedding_mteb_eng_v2_summary.txt",
    )
