from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class WaimaiClustering(AbsTask):
    """Waimai Chinese food delivery review classification dataset."""

    language = "zh"

    hf_name = "sentence-transformers/waimai"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering",
        prompt={
            "query": "Classify food delivery reviews by sentiment"
        },
    )
    
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
