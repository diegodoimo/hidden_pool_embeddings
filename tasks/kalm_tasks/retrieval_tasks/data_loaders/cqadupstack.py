from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CQADupStack(AbsTask):
    """CQADupStack dataset for duplicate question retrieval."""

    language = "en"

    hf_name = "sentence-transformers/cqadupstack"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "duplicate"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve duplicate questions"
        },
    )
    loader = from_one_hf_dataset
