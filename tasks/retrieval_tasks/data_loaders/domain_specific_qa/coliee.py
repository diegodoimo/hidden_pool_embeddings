from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class COLIEE(AbsTask):
    """COLIEE legal case retrieval dataset."""

    language = "en"

    hf_name = "sentence-transformers/coliee"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a legal case, retrieve relevant legal cases"},
    )
    loader = from_one_hf_dataset
