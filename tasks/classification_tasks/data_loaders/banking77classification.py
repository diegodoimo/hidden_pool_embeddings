from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import load_classification_standard


class Banking77Classification(AbsTask):
    hf_name = "mteb/banking77"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["Banking77Classification"]}
    )
    loader = load_classification_standard
