from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.binary_classification_tasks.binary_classification_loaders import (
    load_binary_classification_label_based,
    load_binary_classification_hard_negatives
)


class AmazonCounterfactualClassification(AbsTask):
    language = "en"
    hf_name = "mteb/amazon_counterfactual"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    
    # Label texts for binary classification
    label_texts = {
        0: "not counterfactual",
        1: "counterfactual"
    }
    
    metadata = TaskMetadata(
        type="BinaryClassification", 
        prompt={"query": TASK_PROMPTS["AmazonCounterfactualClassification"]}
    )
    
    # Default to label-based approach
    # Set use_hard_negative_mining = True to use hard negative mining instead
    use_hard_negative_mining = False
    
    @property
    def loader(self):
        if self.use_hard_negative_mining:
            return load_binary_classification_hard_negatives
        else:
            return load_binary_classification_label_based
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate task configuration."""
        pass
