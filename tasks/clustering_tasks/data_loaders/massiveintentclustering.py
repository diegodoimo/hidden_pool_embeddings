from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class MassiveIntentClustering(AbsTask):
    hf_name = "mteb/amazon_massive_intent"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MassiveIntentClassification"]}
    )
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

