from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class AyaDataset(AbsTask):
    """Aya Dataset multilingual instruction-following retrieval."""

    language = "multilingual"

    hf_name = "sentence-transformers/aya-dataset"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant responses"
        },
    )
    loader = from_one_hf_dataset
