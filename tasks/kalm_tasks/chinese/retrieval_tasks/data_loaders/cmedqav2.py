from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CMedQAV2(AbsTask):
    """cMedQA-V2.0 Chinese medical question answering retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/cmedqa-v2"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a medical question, retrieve relevant medical answers"
        },
    )
    loader = from_one_hf_dataset
