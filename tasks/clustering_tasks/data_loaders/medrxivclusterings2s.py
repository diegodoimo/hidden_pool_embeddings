from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class MedrxivClusteringS2S(AbsTask):
    hf_name = "mteb/raw_medrxiv"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MedrxivClusteringS2S"]}
    )
    loader = load_clustering_standard
