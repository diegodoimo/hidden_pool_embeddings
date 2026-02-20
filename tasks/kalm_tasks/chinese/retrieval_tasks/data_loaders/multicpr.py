from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class MultiCPR(AbsTask):
    """Multi-CPR Chinese product retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/multi-cpr"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "product"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant products"
        },
    )
    loader = from_one_hf_dataset
