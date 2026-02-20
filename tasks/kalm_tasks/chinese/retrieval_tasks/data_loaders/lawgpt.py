from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class LawGPT(AbsTask):
    """law-gpt Chinese legal question answering retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/law-gpt"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a legal question, retrieve relevant legal information"
        },
    )
    loader = from_one_hf_dataset
