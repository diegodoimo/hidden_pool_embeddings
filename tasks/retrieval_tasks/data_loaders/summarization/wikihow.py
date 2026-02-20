from tasks.abs_task import AbsTask, TaskMetadata
from tasks.retrieval_loaders import from_one_hf_dataset


class WikiHow(AbsTask):
    """XSum summarization dataset for retrieval (summary -> document)."""

    language = "en"

    hf_name = "gursi26/wikihow-cleaned"
    split = "train"
    has_multiple_datasets = False
    query_name = "summary"
    positive_name = "text"
    title_name = "title"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original document"},
    )
    loader = from_one_hf_dataset
