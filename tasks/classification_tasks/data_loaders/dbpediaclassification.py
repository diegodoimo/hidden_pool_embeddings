from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import load_classification_standard


class DBPediaClassification(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/DBpediaClassification"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": "Identify the category of wiki passages"}
    )
    loader = load_classification_standard
