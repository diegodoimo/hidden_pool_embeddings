from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import sts_preprocessor
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class WikiAnswers(AbsTask):
    """WikiAnswers dataset for paraphrase and semantic textual similarity."""

    language = "en"
    hf_name = "sentence-transformers/wikianswers"
    split = "train"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve paraphrased or similar questions"},
    )
    preprocessor = sts_preprocessor
    loader = from_one_hf_dataset
