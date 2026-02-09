from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class EmotionClustering(AbsTask):
    hf_name = "mteb/emotion"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["EmotionClassification"]}
    )
    loader = load_clustering_standard
