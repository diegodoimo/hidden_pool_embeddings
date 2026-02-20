from datasets import load_dataset
from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class SQuAD(AbsTask):
    """SQuAD dataset for retrieval - question as query, context as positive."""

    language = "en"

    hf_name = "rajpurkar/squad"
    split = "train+validation"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
