from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class MMarcoZh(AbsTask):
    """mMARCO Chinese passage retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/mmarco"
    hf_subset = "zh"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant passages"
        },
    )
    loader = from_one_hf_dataset
