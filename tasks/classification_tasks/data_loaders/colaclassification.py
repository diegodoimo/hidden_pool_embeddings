from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import load_classification_standard


class ColaClassification(AbsTask):
    hf_name = "glue"
    hf_subset = "cola"
    split = "train"
    anchor_name = "sentence"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["ColaClassification"]}
    )
    loader = load_classification_standard
