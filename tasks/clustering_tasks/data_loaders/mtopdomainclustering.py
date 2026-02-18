from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class MTOPDomainClustering(AbsTask):
    hf_name = "mteb/mtop_domain"
    split = "train"
    query_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MTOPDomainClassification"]}
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

