from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class PAWSXZh(AbsTask):
    """PAWS-X Chinese paraphrase adversarial retrieval dataset."""

    language = "zh"

    hf_name = "sentence-transformers/paws-x"
    hf_subset = "zh"
    split = "train"
    has_multiple_datasets = False
    query_name = "sentence1"
    positive_name = "sentence2"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a sentence, retrieve paraphrased sentences"},
    )
    loader = from_one_hf_dataset
