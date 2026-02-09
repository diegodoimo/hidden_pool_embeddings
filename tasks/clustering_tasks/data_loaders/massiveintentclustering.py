from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class MassiveIntentClustering(AbsTask):
    hf_name = "mteb/amazon_massive_intent"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MassiveIntentClassification"]}
    )
    loader = load_clustering_standard
