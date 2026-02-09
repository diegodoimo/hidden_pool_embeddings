from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.clustering_tasks.clustering_loaders import load_clustering_standard


class TweetSentimentExtractionClustering(AbsTask):
    hf_name = "mteb/tweet_sentiment_extraction"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["TweetSentimentExtractionClassification"]}
    )
    loader = load_clustering_standard
