from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class S2ORCTitleAbstract(AbsTask):
    """S2ORC Title-Abstract retrieval dataset."""

    language = "en"

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "title-abstract-pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "title"
    positive_name = "abstract"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Given a paper title, retrieve the abstract"}
    )
    loader = from_one_hf_dataset
