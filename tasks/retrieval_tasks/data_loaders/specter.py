from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class SPECTER(AbsTask):
    """SPECTER scientific paper similarity dataset."""

    hf_name = "sentence-transformers/specter"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["SCIDOCS"]})
    loader = from_one_hf_dataset
