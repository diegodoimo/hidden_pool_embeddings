from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class RedditClusteringS2S(AbsTask):
    hf_name = "sentence-transformers/reddit-title-body"
    split = "train"
    anchor_name = "title"
    label = "subreddit"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["RedditClustering"]}
    )
    loader = load_clustering_standard
