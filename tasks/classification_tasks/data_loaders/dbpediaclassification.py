from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.classification_tasks.classification_loaders import (
    load_multiway_classification_sampling,
    load_multiway_classification_hard_negatives
)


class DBPediaClassification(AbsTask):
    hf_name = "mteb/dbpedia"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": "Identify the category of wiki passages"}
    )
    
    # Use sampling strategy by default
    # Set use_hard_negative_mining = True to use hard negative mining
    use_hard_negative_mining = False
    
    @property
    def loader(self):
        if self.use_hard_negative_mining:
            return load_multiway_classification_hard_negatives
        else:
            return load_multiway_classification_sampling
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate task configuration."""
        pass

    @classmethod
    def validate_config(cls) -> None:
        """Validate task configuration."""
        pass

