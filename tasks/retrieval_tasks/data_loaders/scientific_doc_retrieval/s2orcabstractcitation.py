from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class S2ORCAbstractCitation(AbsTask):
    """S2ORC Abstract-Citation retrieval dataset."""

    language = "en"

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "abstract-citation-pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "abstract"
    positive_name = "citation"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a paper abstract, retrieve abstracts of cited papers"},
    )
    loader = from_one_hf_dataset
