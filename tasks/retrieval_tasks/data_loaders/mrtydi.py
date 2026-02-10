from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_multiple_hf_datasets_with_dedup


class MrTyDi(AbsTask):
    """Mr.TyDi multilingual retrieval with deduplication against MTEB test split."""

    language = "multilingual"

    hf_name = "mteb/mrtydi"
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
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    loader = from_multiple_hf_datasets_with_dedup
