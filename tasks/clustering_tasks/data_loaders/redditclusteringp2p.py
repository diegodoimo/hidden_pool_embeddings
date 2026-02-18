from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class RedditClusteringP2P(AbsTask):
    # ignore_identical_ids = True
    hf_name = "sentence-transformers/reddit-title-body"
    split = "train"
    query_name = "body"
    title_name = "title"
    label_name = "subreddit"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["RedditClusteringP2P"]}
    )


# ===== CLASSIFICATION TASKS =====
    # Use sampling strategy by default
    use_hard_negative_mining = False

    @property
    def loader(self):
        if self.use_hard_negative_mining:
            return load_clustering_hard_negatives
        else:
            return load_clustering_sampling

    @classmethod
    def validate_config(cls) -> None:
        """Validate task configuration."""
        pass

