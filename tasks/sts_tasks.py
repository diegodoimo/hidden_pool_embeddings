from .abs_task import AbsTask, TaskMetadata
import json
from .prompts import QWEN3_PROMPTS as TASK_PROMPTS


class STS12(AbsTask):
    """STS12 Semantic Textual Similarity dataset for retrieval.

    Uses train split to avoid contamination with MTEB evaluation (which uses test).
    """

    hf_name = "mteb/sts12-sts"
    split = "train"  # Use train to avoid MTEB test contamination
    has_multiple_datasets = False
    custom_loader = "load_sts_retrieval"
    anchor_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["STS12"]})


class STS22(AbsTask):
    """STS22 Cross-lingual Semantic Textual Similarity dataset for retrieval.

    Uses train split (all languages) to avoid contamination with MTEB evaluation.
    """

    hf_name = "mteb/sts22-crosslingual-sts"
    hf_subset = None  # Use all languages (default config)
    split = "train"  # Use train to avoid MTEB test contamination
    has_multiple_datasets = False
    custom_loader = "load_sts_retrieval"
    anchor_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["STS22"]})


class STSBenchmark(AbsTask):
    """STS Benchmark dataset for retrieval."""

    hf_name = "mteb/stsbenchmark-sts"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_sts_retrieval"
    anchor_name = "sentence1"
    positive_name = "sentence2"
    score_name = "score"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["STSBenchmark"]}
    )
