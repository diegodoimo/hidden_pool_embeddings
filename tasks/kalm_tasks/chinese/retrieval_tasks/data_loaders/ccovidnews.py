from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class CCOVIDNews(AbsTask):
    """cCOVID-News Chinese COVID-19 news retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/ccovid-news"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query about COVID-19, retrieve relevant news passages"
        },
    )
    loader = from_one_hf_dataset
