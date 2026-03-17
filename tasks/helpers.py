import os
from tasks import (
    NAME_TO_TASK_SUBTASK_PATH,
    TASK_TO_CATEGORY,
    NAME_TO_TASK,
    OPEN_DOMAIN_QA,
    DOMAIN_SPECIFIC_QA,
    GENERAL_RETRIEVAL,
    FACT_VERIFICATION,
    PARAPHRASE_DETECTION,
    SCIENTIFIC_DOC_RETRIEVAL,
    SUMMARIZATION,
    STS_TASKS,
    NLI_TASKS,
    BINARY_CLASSIFICATION_TASKS,
    MULTIWAY_CLASSIFICATION_TASKS,
    CLUSTERING_TASKS,
    DATASETS_BY_SIZE,
)


def task_name_to_inner_path(task_name: str) -> str:
    """Build canonical inner path from NAME_TO_TASK_SUBTASK_PATH.

    Returns e.g. "retrieval/general_retrieval/arguana" or "nli/snli".
    """
    info = NAME_TO_TASK_SUBTASK_PATH[task_name]
    parent = info["parent_folder"]
    subparent = info["subparent_folder"]
    if subparent is not None:
        return os.path.join(parent, subparent, task_name)
    return os.path.join(parent, task_name)


def get_task_category(task_name: str) -> str:
    """
    Get the category for a given task name.

    Args:
        task_name: Name of the task (lowercase)

    Returns:
        Category path (e.g., "retrieval/open_domain_qa", "nli", "sts"), or None if unknown
    """
    return TASK_TO_CATEGORY.get(task_name.lower(), None)


def get_category_path(task_name: str, base_path: str) -> str:
    """
    Get the full path for saving a task's dataset, organized by category.

    Args:
        task_name: Name of the task
        base_path: Base path for datasets (e.g., "./results/datasets_negatives/model_name")

    Returns:
        Full path including category subfolder(s) if applicable, otherwise base path with task name

    Examples:
        >>> get_category_path("msmarco", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/retrieval/general_retrieval/msmarco"

        >>> get_category_path("snli", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/nli/snli"

        >>> get_category_path("xnli", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/nli/xnli"
    """
    category = get_task_category(task_name)
    if category:
        return f"{base_path}/{category}/{task_name}", category
    else:
        # Unknown task, keep in flat structure
        return f"{base_path}/{task_name}", "category_unknown"


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
        selected_tasks.extend(MULTIWAY_CLASSIFICATION_TASKS)

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


def get_task(name: str):
    if name not in NAME_TO_TASK:
        raise ValueError(
            f"Unknown task '{name}'. Available tasks: {list(NAME_TO_TASK)}"
        )

    task = NAME_TO_TASK[name]

    return task
