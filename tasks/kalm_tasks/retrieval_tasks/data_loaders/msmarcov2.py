from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_multiple_hf_datasets


class MSMARCOv2(AbsTask):
    language = "en"
    hf_name = "mteb/msmarco-v2"
    split = "train"
    has_multiple_datasets = True
    query_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "query_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["MSMARCO"]})
    loader = from_multiple_hf_datasets
