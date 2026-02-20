from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_multiple_hf_datasets


class FiQA2018(AbsTask):
    """FiQA 2018 financial QA dataset with deduplication against MTEB test split."""

    language = "en"

    hf_name = "mteb/fiqa"
    split = "train+dev"
    has_multiple_datasets = True
    eval_split = "test"  # MTEB evaluates on test split
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
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["FiQA2018"]}
    )
    loader = from_multiple_hf_datasets
