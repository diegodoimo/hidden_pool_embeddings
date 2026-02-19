from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import nli_preprocessor
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class CAIL2019SCM(AbsTask):
    """CAIL2019-SCM Chinese legal case matching dataset."""

    language = "zh"

    hf_name = "sentence-transformers/cail2019-scm"
    split = "train"
    has_multiple_datasets = False
    query_name = "premise"
    positive_name = "hypothesis"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a legal case, retrieve similar legal cases"},
    )
    preprocessor = nli_preprocessor
    loader = from_one_hf_dataset
