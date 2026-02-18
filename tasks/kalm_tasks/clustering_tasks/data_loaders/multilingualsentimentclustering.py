from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import (
    load_clustering_sampling,
    load_clustering_hard_negatives
)


class MultilingualSentimentClustering(AbsTask):
    """MultilingualSentiment multilingual sentiment classification dataset."""

    language = "multilingual"

    hf_name = "sentence-transformers/multilingual-sentiment"
    split = "train"
    has_multiple_datasets = False
    query_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering",
        prompt={
            "query": "Classify text by sentiment"
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
