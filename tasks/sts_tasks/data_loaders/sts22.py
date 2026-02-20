from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_helpers import sts_preprocessor
from tasks.retrieval_loaders import from_one_hf_dataset


class STS22(AbsTask):
    """STS22 Semantic Textual Similarity dataset for retrieval.

    Uses train split to avoid contamination with MTEB evaluation (which uses test).
    """

    language = "en"

    hf_name = "mteb/sts22-crosslingual-sts"
    hf_subset = None  # Use all languages (default config)
    split = "train"  # Use train to avoid MTEB test contamination
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["STS22"]})
    preprocessor = sts_preprocessor
    loader = from_one_hf_dataset
