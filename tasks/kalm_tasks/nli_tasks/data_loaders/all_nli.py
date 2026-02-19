from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class ALL_NLI(AbsTask):
    language = "en"
    hf_name = "sentence-transformers/all-nli"
    hf_subset = "triplet"
    split = "train+dev"
    has_multiple_datasets = False
    query_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Retrieve semantically similar text"}
    )
    loader = from_one_hf_dataset
