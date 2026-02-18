from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CodeFeedback(AbsTask):
    """CodeFeedback dataset for code-related retrieval."""

    language = "en"

    hf_name = "sentence-transformers/codefeedback"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a code question, retrieve passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
