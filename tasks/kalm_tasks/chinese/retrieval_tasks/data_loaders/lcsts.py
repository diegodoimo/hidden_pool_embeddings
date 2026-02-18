from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class LCSTS(AbsTask):
    """LCSTS Chinese short text summarization retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/lcsts"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "text"
    positive_name = "summary"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a text, retrieve its summary"
        },
    )
    loader = from_one_hf_dataset
