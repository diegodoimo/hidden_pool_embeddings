from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import load_classification_standard


class ToxicConversations50k(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/toxic_conversations_50k"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["ToxicConversationsClassification"]}
    )
    loader = load_classification_standard
