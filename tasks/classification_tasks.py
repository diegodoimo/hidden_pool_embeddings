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
