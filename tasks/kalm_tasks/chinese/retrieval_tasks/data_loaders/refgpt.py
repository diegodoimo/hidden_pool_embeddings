from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class RefGPT(AbsTask):
    """RefGPT Chinese reference-based retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/refgpt"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant passages"
        },
    )
    loader = from_one_hf_dataset
