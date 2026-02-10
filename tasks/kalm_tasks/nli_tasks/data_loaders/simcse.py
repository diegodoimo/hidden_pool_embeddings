from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_nli_retrieval


class SimCSENLI(AbsTask):
    """SimCSE NLI dataset for natural language inference."""

    language = "en"

    hf_name = "sentence-transformers/simcse-nli"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # Assuming 0 = entailment
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval
