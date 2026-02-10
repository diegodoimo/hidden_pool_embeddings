from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class UMETRIPQA(AbsTask):
    """UMETRIP-QA Chinese travel question answering retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/umetrip-qa"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a travel question, retrieve relevant answers"
        },
    )
    loader = from_one_hf_dataset
