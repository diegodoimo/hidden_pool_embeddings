from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class QQP(AbsTask):
    """Quora Question Pairs dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/quora-duplicates"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "anchor"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["QuoraRetrieval"]}
    )
    loader = from_one_hf_dataset
