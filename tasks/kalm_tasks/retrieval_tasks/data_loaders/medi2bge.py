from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class MEDI2BGE(AbsTask):
    """MEDI2BGE dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/medi2bge"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a query, retrieve relevant passages"},
    )
    loader = from_one_hf_dataset
