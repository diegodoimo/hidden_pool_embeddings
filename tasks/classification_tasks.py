from .abs_task import AbsTask, TaskMetadata
import json
from .prompts import QWEN3_PROMPTS as TASK_PROMPTS


class DBPediaClassification(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/DBpediaClassification"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": "Identify the category of wiki passages"}
    )


class ToxicConversations50k(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/toxic_conversations_50k"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["ToxicConversationsClassification"]}
    )


class Banking77Classification(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/banking77"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["Banking77Classification"]}
    )


class RedditClusteringP2P(AbsTask):
    # ignore_identical_ids = True
    hf_name = "sentence-transformers/reddit-title-body"
    split = "train"
    anchor_name = "body"
    title_name = "title"
    label_name = "subreddit"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["RedditClusteringP2P"]}
    )


# ===== CLASSIFICATION TASKS =====

class AmazonCounterfactualClassification(AbsTask):
    hf_name = "mteb/amazon_counterfactual"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["AmazonCounterfactualClassification"]}
    )


class AmazonPolarityClassification(AbsTask):
    hf_name = "mteb/amazon_polarity"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["AmazonPolarityClassification"]}
    )


class ImdbClassification(AbsTask):
    hf_name = "mteb/imdb"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["ImdbClassification"]}
    )


class ColaClassification(AbsTask):
    hf_name = "glue"
    hf_subset = "cola"
    split = "train"
    anchor_name = "sentence"
    label = "label"
    metadata = TaskMetadata(
        type="Classification", prompt={"query": TASK_PROMPTS["ColaClassification"]}
    )


# ===== CLUSTERING TASKS =====

class AmazonReviewsClustering(AbsTask):
    hf_name = "mteb/amazon_reviews_multi"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "product_category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["AmazonReviewsClassification"]}
    )


class EmotionClustering(AbsTask):
    hf_name = "mteb/emotion"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["EmotionClassification"]}
    )


class MTOPIntentClustering(AbsTask):
    hf_name = "mteb/mtop_intent"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MTOPIntentClassification"]}
    )


class MTOPDomainClustering(AbsTask):
    hf_name = "mteb/mtop_domain"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MTOPDomainClassification"]}
    )


class MassiveScenarioClustering(AbsTask):
    hf_name = "mteb/amazon_massive_scenario"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MassiveScenarioClassification"]}
    )


class MassiveIntentClustering(AbsTask):
    hf_name = "mteb/amazon_massive_intent"
    hf_subset = "en"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MassiveIntentClassification"]}
    )


class TweetSentimentExtractionClustering(AbsTask):
    hf_name = "mteb/tweet_sentiment_extraction"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["TweetSentimentExtractionClassification"]}
    )


class ArxivClusteringP2P(AbsTask):
    hf_name = "mteb/raw_arxiv"
    split = "train"
    anchor_name = "abstract"
    title_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["ArxivClusteringP2P"]}
    )


class ArxivClusteringS2S(AbsTask):
    hf_name = "mteb/raw_arxiv"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["ArxivClusteringS2S"]}
    )


class BiorxivClusteringP2P(AbsTask):
    hf_name = "mteb/raw_biorxiv"
    split = "train"
    anchor_name = "abstract"
    title_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["BiorxivClusteringP2P"]}
    )


class BiorxivClusteringS2S(AbsTask):
    hf_name = "mteb/raw_biorxiv"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["BiorxivClusteringS2S"]}
    )


class MedrxivClusteringP2P(AbsTask):
    hf_name = "mteb/raw_medrxiv"
    split = "train"
    anchor_name = "abstract"
    title_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MedrxivClusteringP2P"]}
    )


class MedrxivClusteringS2S(AbsTask):
    hf_name = "mteb/raw_medrxiv"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["MedrxivClusteringS2S"]}
    )


class RedditClusteringS2S(AbsTask):
    hf_name = "sentence-transformers/reddit-title-body"
    split = "train"
    anchor_name = "title"
    label = "subreddit"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["RedditClustering"]}
    )


class StackExchangeClusteringP2P(AbsTask):
    hf_name = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
    split = "train"
    anchor_name = "body"
    title_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["StackExchangeClusteringP2P"]}
    )


class StackExchangeClusteringS2S(AbsTask):
    hf_name = "flax-sentence-embeddings/stackexchange_title_body_jsonl"
    split = "train"
    anchor_name = "title"
    label = "category"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["StackExchangeClustering"]}
    )


class TwentyNewsgroupsClustering(AbsTask):
    hf_name = "SetFit/20_newsgroups"
    split = "train"
    anchor_name = "text"
    label = "label"
    metadata = TaskMetadata(
        type="Clustering", prompt={"query": TASK_PROMPTS["TwentyNewsgroupsClustering"]}
    )
