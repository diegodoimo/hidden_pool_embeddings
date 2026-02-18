from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class ESCI(AbsTask):
    """ESCI dataset for e-commerce product retrieval."""

    language = "en"

    hf_name = "sentence-transformers/esci"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "product"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a product search query, retrieve relevant products"
        },
    )
    loader = from_one_hf_dataset
