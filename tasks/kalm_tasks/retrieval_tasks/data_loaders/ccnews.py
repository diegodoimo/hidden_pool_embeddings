from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CCNews(AbsTask):
    """CC-News dataset for news article retrieval."""

    language = "en"

    hf_name = "sentence-transformers/cc-news"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant news passages"
        },
    )
    loader = from_one_hf_dataset
