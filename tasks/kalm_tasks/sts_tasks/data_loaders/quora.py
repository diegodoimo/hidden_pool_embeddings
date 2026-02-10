from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import load_sts_retrieval


class Quora(AbsTask):
    """Quora duplicate questions dataset for semantic textual similarity."""

    language = "en"
    hf_name = "sentence-transformers/quora-duplicates"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve duplicate or similar questions"
        },
    )
    loader = load_sts_retrieval
