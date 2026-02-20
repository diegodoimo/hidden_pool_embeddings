from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.sts_tasks.sts_helpers import sts_preprocessor
from tasks.retrieval_loaders import from_one_hf_dataset


class STSBenchmark(AbsTask):
    """STSBenchmark Semantic Textual Similarity dataset for retrieval.

    Uses train split to avoid contamination with MTEB evaluation (which uses test).
    """

    language = "en"

    hf_name = "mteb/stsbenchmark-sts"
    split = "train+validation"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["STSBenchmark"]}
    )
    preprocessor = sts_preprocessor
    loader = from_one_hf_dataset
