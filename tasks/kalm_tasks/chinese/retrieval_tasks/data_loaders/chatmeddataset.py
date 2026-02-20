from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class ChatMedDataset(AbsTask):
    """ChatMed-Dataset Chinese medical question answering retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/chatmed-dataset"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a medical query, retrieve relevant medical information"
        },
    )
    loader = from_one_hf_dataset
