from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class LIMAZH(AbsTask):
    """LIMA Chinese instruction-following dataset for retrieval."""

    language = "zh"

    hf_name = "sentence-transformers/lima"
    hf_subset = "zh"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a query, retrieve relevant responses"},
    )
    loader = from_one_hf_dataset
