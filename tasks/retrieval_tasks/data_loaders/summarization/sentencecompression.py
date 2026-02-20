from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class SentenceCompression(AbsTask):
    """Sentence Compression dataset for retrieval."""

    language = "en"

    hf_name = "sentence-transformers/sentence-compression"
    split = "train"
    has_multiple_datasets = False
    query_name = "text"
    positive_name = "simplified"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a compressed sentence, retrieve the original sentence"},
    )
    loader = from_one_hf_dataset
