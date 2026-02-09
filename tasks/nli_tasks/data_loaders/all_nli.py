from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_all_nli_retrieval


class ALL_NLI(AbsTask):
    hf_name = "sentence-transformers/all-nli"
    hf_subset = "triplet"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Retrieve semantically similar text"}
    )
    loader = load_all_nli_retrieval
