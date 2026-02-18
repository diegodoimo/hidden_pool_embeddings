from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import load_sts_retrieval


class NLLB(AbsTask):
    """NLLB dataset for semantic textual similarity."""

    language = "multilingual"

    hf_name = "sentence-transformers/nllb"
    hf_subset = "en"
    split = "train"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a sentence, retrieve semantically similar sentences"
        },
    )
    loader = load_sts_retrieval
