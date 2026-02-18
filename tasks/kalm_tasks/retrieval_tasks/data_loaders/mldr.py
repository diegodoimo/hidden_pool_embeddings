from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class MLDR(AbsTask):
    """MLDR (Multilingual Long Document Retrieval) dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/mldr"
    hf_subset = "en"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant documents"
        },
    )
    loader = from_one_hf_dataset
