from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_nli_retrieval


class ANLI(AbsTask):
    """ANLI dataset for retrieval - premise as query, entailed hypothesis as positive."""

    hf_name = "facebook/anli"
    split = "train_r1"  # Can also use train_r2, train_r3
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # 0 = entailment in ANLI
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval
