from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class TwentyNewsgroupsClustering(AbsTask):
    hf_name = "SetFit/20_newsgroups"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["TwentyNewsgroupsClustering"]}
    )
    loader = load_clustering_standard
