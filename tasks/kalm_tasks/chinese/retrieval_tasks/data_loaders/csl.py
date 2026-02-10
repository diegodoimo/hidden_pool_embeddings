from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CSL(AbsTask):
    """CSL Chinese Scientific Literature retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/csl"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant scientific literature"
        },
    )
    loader = from_one_hf_dataset
