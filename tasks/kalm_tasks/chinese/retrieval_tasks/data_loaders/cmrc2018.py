from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CMRC2018(AbsTask):
    """CMRC 2018 Chinese machine reading comprehension retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/cmrc2018"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
