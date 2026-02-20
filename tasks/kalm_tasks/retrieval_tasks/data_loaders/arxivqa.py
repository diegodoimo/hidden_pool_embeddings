from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class ArxivQA(AbsTask):
    """arXiv QA dataset for scientific paper retrieval."""

    language = "en"

    hf_name = "sentence-transformers/arxiv_qa"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question about scientific papers, retrieve relevant passages"
        },
    )
    loader = from_one_hf_dataset
