from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_nli_retrieval


class CAIL2019SCM(AbsTask):
    """CAIL2019-SCM Chinese legal case matching dataset."""

    language = "zh"

    hf_name = "sentence-transformers/cail2019-scm"
    split = "train"
    has_multiple_datasets = False
    query_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a legal case, retrieve similar legal cases"
        },
    )
    loader = load_nli_retrieval
