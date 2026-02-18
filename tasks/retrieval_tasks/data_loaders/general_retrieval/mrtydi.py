from tasks.abs_task import AbsTask, TaskMetadata
from tasks.retrieval_tasks.retrieval_loaders import from_multiple_hf_datasets


MRTYDI_SUBTASKS = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]


class MrTyDi(AbsTask):
    """Mr.TyDi multilingual retrieval dataset.

    Each language is loaded as a separate subtask.
    The HF subset names are constructed as ``{subtask}-queries``,
    ``{subtask}-corpus``, ``{subtask}-qrels`` (e.g. ``arabic-queries``).
    """

    language = "multilingual"

    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "queries"
    positive_name = "corpus"
    qrels_name = "qrels"
    qrels_fields = {
        "query_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    subtasks = MRTYDI_SUBTASKS
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    loader = from_multiple_hf_datasets
