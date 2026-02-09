from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import load_classification_standard


class AmazonCounterfactualClassification(AbsTask):
    hf_name = "mteb/amazon_counterfactual"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["AmazonCounterfactualClassification"]}
    )
    loader = load_classification_standard
