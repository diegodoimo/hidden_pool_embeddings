from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import load_sts_retrieval


class WikiAnswers(AbsTask):
    """WikiAnswers dataset for paraphrase and semantic textual similarity."""

    language = "en"
    hf_name = "sentence-transformers/wikianswers"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve paraphrased or similar questions"
        },
    )
    loader = load_sts_retrieval
