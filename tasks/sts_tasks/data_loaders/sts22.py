from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_loaders import load_sts_retrieval


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
    score_name = "score"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["STS22"]})
    loader = load_sts_retrieval
