from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class ExpertQA(AbsTask):
    """ExpertQA dataset for expert-level question answering retrieval."""

    language = "en"

    hf_name = "sentence-transformers/expertqa"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that provide expert answers"
        },
    )
    loader = from_one_hf_dataset
