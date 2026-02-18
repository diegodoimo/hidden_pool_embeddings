from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CHEF(AbsTask):
    """CHEF Chinese dataset for health and food retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/chef"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query about health and food, retrieve relevant passages"
        },
    )
    loader = from_one_hf_dataset
