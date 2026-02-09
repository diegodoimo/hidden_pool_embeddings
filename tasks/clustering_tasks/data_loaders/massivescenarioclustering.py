from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class MassiveScenarioClustering(AbsTask):
    hf_name = "mteb/amazon_massive_scenario"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MassiveScenarioClassification"]}
    )
    loader = load_clustering_standard
