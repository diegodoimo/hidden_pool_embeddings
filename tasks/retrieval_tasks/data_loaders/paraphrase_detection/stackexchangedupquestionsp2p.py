from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class StackExchangeDupQuestionsP2P(AbsTask):
    """StackExchange duplicate questions (post to post)."""

    language = "en"

    hf_name = "sentence-transformers/stackexchange-duplicates"
    hf_subset = "post-post-pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "post1"
    positive_name = "post2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question post, retrieve duplicate question posts"},
    )
    loader = from_one_hf_dataset
