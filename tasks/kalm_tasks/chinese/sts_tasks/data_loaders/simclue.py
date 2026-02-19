from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import sts_preprocessor
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class SimCLUE(AbsTask):
    """SimCLUE Chinese semantic textual similarity dataset."""

    language = "zh"

    hf_name = "sentence-transformers/simclue"
    split = "train"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a sentence, retrieve semantically similar sentences"},
    )
    preprocessor = sts_preprocessor
    loader = from_one_hf_dataset
