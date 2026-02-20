from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class S2ORCTitleCitation(AbsTask):
    """S2ORC Title-Citation retrieval dataset."""

    language = "en"

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "title-citation-pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "title"
    positive_name = "citation"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a paper title, retrieve titles of cited papers"},
    )
    loader = from_one_hf_dataset
