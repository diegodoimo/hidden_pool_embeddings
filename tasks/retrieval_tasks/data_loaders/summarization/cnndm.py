from tasks.abs_task import AbsTask, TaskMetadata
from tasks.retrieval_loaders import from_one_hf_dataset


class CNNDM(AbsTask):
    """CNN/DailyMail summarization dataset for retrieval (highlights -> article)."""

    language = "en"

    hf_name = "abisee/cnn_dailymail"
    hf_subset = "3.0.0"
    split = "train+validation"
    has_multiple_datasets = False
    query_name = "highlights"
    positive_name = "article"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original article"},
    )
    loader = from_one_hf_dataset
