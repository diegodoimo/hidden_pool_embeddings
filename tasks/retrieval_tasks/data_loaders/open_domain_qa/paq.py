from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class PAQ(AbsTask):
    """PAQ (Probably Asked Questions) dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/paq"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
