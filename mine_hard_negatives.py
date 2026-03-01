from inference.hard_negative_mining import HardNegativesMiner
import os
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from utils.create_datasets import instruction_template_qwen3
from datetime import timedelta
from tasks import (
    NAME_TO_TASK,
    BINARY_CLASSIFICATION_TASKS,
    CLASSIFICATION_TASKS,
    NLI_TASKS,
    STS_TASKS,
    CLUSTERING_TASKS,
    DATASETS_BY_SIZE,
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

from utils.helpers import print_memory_consumed
from models.modules import add_pooling_layers, last_token_pool


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
        default=None,
        help="Select task types to mine hard negatives for. Can specify multiple types. Ignored if --task_names is provided.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
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

    if "sorted" in task_types:
        selected_tasks.extend(DATASETS_BY_SIZE)

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

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    dist.init_process_group(
        "nccl",
        device_id=LOCAL_RANK,
        timeout=timedelta(seconds=600),
    )
    torch.cuda.set_device(dist.get_rank())

    # enable tensorfloat32
    torch.set_float32_matmul_precision("high")

    # Select tasks based on task_names (if provided) or task_types
    selected_tasks = validate_and_select_tasks(args.task_names, args.task_types)

    if RANK == 0:
        if args.task_names is not None:
            print(f"Selected specific tasks: {args.task_names}")
        else:
            print(f"Selected task types: {args.task_types}")
        print(f"Tasks to process: {selected_tasks}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    max_length = min(args.max_length, model.config.max_position_embeddings)

    miner = HardNegativesMiner(
        path=f"./results/datasets_negatives/{path_to_name[args.model_name_or_path]}",
        model_name=path_to_name[args.model_name_or_path],
        task_names=selected_tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template_qwen3,
        padding_side="right",
        max_length=max_length,
        add_special_tokens=False,
        eot_id=tokenizer.pad_token_id,
        iterative_encode_threshold=10**7
    )

    if RANK == 0:
        print("model loaded")
    dist.barrier()
    model = model.eval()
    # ddp is only needed for training here we are adding gradient buffers and the memory occupied with doubl
    # model = DDP(model, device_ids=[dist.get_rank()])
    model = torch.compile(model)
    model = add_pooling_layers(model, pool_fn=last_token_pool)

    if RANK == 0:
        print("model wrapped in DDP and compile")
        print_memory_consumed()
    dist.barrier()

    miner.mine_negatives(model, batch_size=args.batch_size)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
