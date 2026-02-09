from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class StackExchangeClusteringS2S(AbsTask):
    hf_name = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["StackExchangeClustering"]}
    )
    loader = load_clustering_standard
