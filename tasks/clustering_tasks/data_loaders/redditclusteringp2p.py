from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class RedditClusteringP2P(AbsTask):
    # ignore_identical_ids = True
    hf_name = "sentence-transformers/reddit-title-body"
    split = "train"
    anchor_name = "body"
    title_name = "title"
    label_name = "subreddit"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["RedditClusteringP2P"]}
    )


# ===== CLASSIFICATION TASKS =====
    loader = load_clustering_standard
