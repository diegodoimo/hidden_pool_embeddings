from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class BiorxivClusteringP2P(AbsTask):
    hf_name = "mteb/raw_biorxiv"
    split = "train"
    anchor_name = "abstract"
    title_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["BiorxivClusteringP2P"]}
    )
    loader = load_clustering_standard
