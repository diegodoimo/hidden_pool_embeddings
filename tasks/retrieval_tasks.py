from .abs_task import AbsTask, TaskMetadata
import json
from .prompts import QWEN3_PROMPTS as TASK_PROMPTS


class MSMARCO(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/msmarco"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {"anchor_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["MSMARCO"]})


class MSMARCOv2(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/msmarco-v2"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {"anchor_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["MSMARCO"]})


class NFCorpus(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/nfcorpus"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {"anchor_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["NFCorpus"]})


class FEVER(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/fever"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {"anchor_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["FEVER"]})


class HotpotQA(AbsTask):
    # ignore_identical_ids = True
    hf_name = "mteb/hotpotqa"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {"anchor_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["HotpotQA"]})


class NaturalQuestions(AbsTask):
    # ignore_identical_ids = True
    hf_name = "sentence-transformers/natural-questions"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["NQ"]})


class ALL_NLI(AbsTask):
    # ignore_identical_ids = True
    hf_name = "sentence-transformers/all-nli"
    hf_subset_name = "triplet"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Retrieve semantically similar text"}
    )
