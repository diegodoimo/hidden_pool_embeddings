from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_nli_retrieval


class MNLI(AbsTask):
    """MNLI dataset for retrieval - premise as query, entailed hypothesis as positive."""

    language = "en"

    hf_name = "nyu-mll/multi_nli"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # 0 = entailment in MNLI
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval
