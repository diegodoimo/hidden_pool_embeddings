from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class AmazonQA(AbsTask):
    """Amazon QA dataset for retrieval - question as query, answer as positive."""

    hf_name = "sentence-transformers/amazon-qa"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a product question, retrieve relevant answers from Amazon product pages"
        },
    )
    loader = from_one_hf_dataset
