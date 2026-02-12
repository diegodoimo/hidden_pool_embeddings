from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class TriviaQA(AbsTask):
    """TriviaQA dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/trivia-qa"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a trivia question, retrieve the answer"},
    )
    loader = from_one_hf_dataset
