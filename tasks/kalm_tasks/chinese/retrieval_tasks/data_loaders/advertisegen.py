from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class AdvertiseGen(AbsTask):
    """AdvertiseGen Chinese dataset for advertising generation retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/advertisegen"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant advertising content"
        },
    )
    loader = from_one_hf_dataset
