from tasks.abs_task import AbsTask, TaskMetadata
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class XSum(AbsTask):
    """XSum summarization dataset for retrieval (summary -> document)."""

    language = "en"

    hf_name = "EdinburghNLP/xsum"
    split = "train+validation"
    has_multiple_datasets = False
    query_name = "summary"
    positive_name = "document"
    corpus_fields = None
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original document"},
    )
    loader = from_one_hf_dataset
