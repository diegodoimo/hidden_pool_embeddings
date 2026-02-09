from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_multiple_hf_datasets_with_dedup


class NFCorpus(AbsTask):
    """NFCorpus with deduplication against MTEB test split."""
    hf_name = "mteb/nfcorpus"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["NFCorpus"]}
    )
    loader = from_multiple_hf_datasets_with_dedup
