from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_helpers import sts_preprocessor
from tasks.retrieval_loaders import from_one_hf_dataset


class Quora(AbsTask):
    """Quora duplicate questions dataset for semantic textual similarity."""

    language = "en"
    hf_name = "sentence-transformers/quora-duplicates"
    split = "train"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve duplicate or similar questions"},
    )
    preprocessor = sts_preprocessor
    loader = from_one_hf_dataset
