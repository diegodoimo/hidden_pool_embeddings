from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class StackExchangeDupQuestionsS2S(AbsTask):
    """StackExchange duplicate questions (title to title)."""

    language = "en"

    hf_name = "sentence-transformers/stackexchange-duplicates"
    hf_subset = "title-title-pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "title1"
    positive_name = "title2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question title, retrieve duplicate question titles"},
    )
    loader = from_one_hf_dataset
