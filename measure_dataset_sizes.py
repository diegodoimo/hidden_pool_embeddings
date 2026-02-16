import argparse
import os
import torch
import torch.distributed as dist

from matplotlib import category
from tasks import (
    NAME_TO_TASK,
    BINARY_CLASSIFICATION_TASKS,
    CLASSIFICATION_TASKS,
    NLI_TASKS,
    STS_TASKS,
    CLUSTERING_TASKS,
    get_task,
)
from tasks.task_categories import (
    OPEN_DOMAIN_QA,
    DOMAIN_SPECIFIC_QA,
    GENERAL_RETRIEVAL,
    FACT_VERIFICATION,
    PARAPHRASE_DETECTION,
    SCIENTIFIC_DOC_RETRIEVAL,
    SUMMARIZATION,
)

from tasks.task_categories import get_category_path
from tasks.load_datasets import load_task_data
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific task names to mine hard negatives for (e.g., 'msmarco' 'hotpotqa'). Takes precedence over --task_types.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        choices=["retrieval", "sts", "nli", "classification", "clustering", "all"],
        default=None,
        help="Select task types to mine hard negatives for. Can specify multiple types. Ignored if --task_names is provided.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--path", type=str, default="results/datasets")
    parser.add_argument(
        "--filename",
        type=str,
        help="filename to save the statistics",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of statistics even if they exist",
    )
    args = parser.parse_args()
    return args


path_to_name = {
    "Qwen/Qwen3-Embedding-0.6B": "qwen3_600m",
    "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
}


# All other retrieval tasks (everything not in the above categories)
def get_retrieval_tasks():
    """
    Get all retrieval tasks (excluding STS, NLI, classification, and clustering tasks),
    sorted by their category group in a consistent order.

    Returns tasks in this order:
    1. Open Domain QA
    2. Domain-Specific QA
    3. General Retrieval
    4. Fact Verification
    5. Paraphrase Detection
    6. Scientific Document Retrieval
    7. Summarization
    """
    # Define the order of retrieval task categories
    retrieval_categories = [
        OPEN_DOMAIN_QA,
        DOMAIN_SPECIFIC_QA,
        GENERAL_RETRIEVAL,
        FACT_VERIFICATION,
        PARAPHRASE_DETECTION,
        SCIENTIFIC_DOC_RETRIEVAL,
        SUMMARIZATION,
    ]

    # Build sorted list of retrieval tasks by category
    sorted_tasks = []
    for category in retrieval_categories:
        sorted_tasks.extend(category)

    return sorted_tasks


def filter_tasks_by_type(task_types):
    """
    Filter available tasks based on requested task types.

    Args:
        task_types: List of task type strings ("retrieval", "sts", "nli", "classification", "clustering", "all")

    Returns:
        List of task names matching the requested types
    """
    if "all" in task_types:
        return list(NAME_TO_TASK.keys())

    selected_tasks = []

    if "retrieval" in task_types:
        selected_tasks.extend(get_retrieval_tasks())

    if "sts" in task_types:
        selected_tasks.extend(STS_TASKS)

    if "nli" in task_types:
        selected_tasks.extend(NLI_TASKS)

    if "classification" in task_types:
        selected_tasks.extend(BINARY_CLASSIFICATION_TASKS)
        selected_tasks.extend(CLASSIFICATION_TASKS)

    if "clustering" in task_types:
        selected_tasks.extend(CLUSTERING_TASKS)

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for task in selected_tasks:
        if task not in seen and task in NAME_TO_TASK:
            seen.add(task)
            result.append(task)

    return result


def validate_and_select_tasks(task_names, task_types):
    """
    Validate and select tasks based on task_names or task_types.

    Args:
        task_names: List of specific task names or None
        task_types: List of task type strings

    Returns:
        List of validated task names

    Raises:
        ValueError: If any task name is invalid
    """
    if task_names is not None:
        # Use specific task names if provided
        invalid_tasks = [task for task in task_names if task not in NAME_TO_TASK]
        if invalid_tasks:
            available_tasks = sorted(NAME_TO_TASK.keys())
            raise ValueError(
                f"Invalid task name(s): {invalid_tasks}\n"
                f"Available tasks: {available_tasks}"
            )
        return task_names
    else:
        # Fall back to task types
        return filter_tasks_by_type(task_types)


def main():
    args = parse_args()

    # Initialize PyTorch distributed (required by load_task_data)
    # Use gloo backend for CPU-only operation
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    # Get rank for distributed processing
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Initialized distributed with {world_size} process(es)")

    # Select tasks based on task_names (if provided) or task_types
    task_names = validate_and_select_tasks(args.task_names, args.task_types)
    if rank == 0:
        if args.task_names is not None:
            print(f"Selected specific tasks: {args.task_names}")
        else:
            print(f"Selected task types: {args.task_types}")
        print(f"Tasks to process: {task_names}")

    # Load existing stats if file exists (only on rank 0)
    stats = {}
    if rank == 0:
        output_file = Path(args.path) / args.filename

        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            with open(output_file, "r") as f:
                stats = json.load(f)
            print(
                f"Loaded existing statistics for {len(stats)} tasks from {output_file}"
            )
    else:
        output_file = None

    for task_name in task_names:
        if rank == 0:
            if task_name in stats and not args.force_recompute:
                print(
                    f"Skipping {task_name}, already in stats. Use --force_recompute to override."
                )
                continue

        try:
            _, category = get_category_path(task_name, args.path)
            task_type, category_name = category.split("/")

            if rank == 0:
                print(f"\n\nPREPARING DATASET {category}: {task_name}\n")

            task = get_task(task_name)
            subtasks = getattr(task, "subtasks", None)
            has_subtasks = subtasks is not None
            if not has_subtasks:
                subtasks = [None]

            task_stats = {
                "task_type": task_type,
                "category": category_name,
                "subsets": {} if has_subtasks else None,
            }

            accumulated_stats = {
                "total_queries": 0,
                "unique_queries": 0,
                "unique_positives": 0,
                "unique_documents": 0,
                "total_qrels": 0,
            }

            for subtask in subtasks:
                if rank == 0 and subtask is not None:
                    print(f"  Processing subtask: {subtask}")

                loaded_data = load_task_data(
                    task, subtask=subtask, max_num_queries=None
                )

                if not isinstance(loaded_data, tuple):
                    if rank == 0:
                        print(
                            f"  Skipping non-retrieval data for {task_name}"
                            + (f" subtask {subtask}" if subtask else "")
                        )
                    continue

                # Retrieval/STS task: (hf_dataset, corpus_dict, query_dict, has_title, n_positives)
                data_split, corpus_dict, query_dict, has_title, n_positives = (
                    loaded_data
                )

                subtask_stats = {
                    "total_queries": len(data_split["qrels"]),
                    "unique_queries": len(data_split["unique_queries"]),
                    "unique_positives": int(n_positives),
                    "unique_documents": len(data_split["corpus"]),
                    "total_qrels": len(data_split["qrels"]),
                }

                accumulated_stats["total_queries"] += subtask_stats["total_queries"]
                accumulated_stats["unique_queries"] += subtask_stats["unique_queries"]
                accumulated_stats["unique_positives"] += subtask_stats[
                    "unique_positives"
                ]
                accumulated_stats["unique_documents"] += subtask_stats[
                    "unique_documents"
                ]
                accumulated_stats["total_qrels"] += subtask_stats["total_qrels"]

                if has_subtasks:
                    task_stats["subsets"][subtask] = subtask_stats

            # Populate task_stats with aggregated values
            task_stats.update(accumulated_stats)

            stats[task_name] = task_stats

            if rank == 0:
                with open(output_file, "w") as f:
                    json.dump(stats, f, indent=2)
                    print(f"Updated statistics saved to {output_file}")

        except Exception as e:
            if rank == 0:
                print(
                    f"Skipping {task_name} due to error: {type(e).__name__}: {str(e)}"
                )
            continue

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
