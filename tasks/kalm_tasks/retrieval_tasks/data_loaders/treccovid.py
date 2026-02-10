from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class TRECCOVID(AbsTask):
    """TREC-COVID dataset for COVID-19 scientific article retrieval."""

    language = "en"

    hf_name = "sentence-transformers/trec-covid"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query about COVID-19, retrieve relevant scientific passages"
        },
    )
    loader = from_one_hf_dataset
