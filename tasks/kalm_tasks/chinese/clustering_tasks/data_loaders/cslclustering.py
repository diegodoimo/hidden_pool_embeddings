from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class CSLClustering(AbsTask):
    """CSL Chinese Scientific Literature classification dataset."""

    language = "zh"

    hf_name = "sentence-transformers/csl"
    hf_subset = "classification"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering",
        prompt={
            "query": "Classify scientific literature by category"
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
