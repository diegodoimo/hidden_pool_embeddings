from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import nli_preprocessor
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class SimCSENLI(AbsTask):
    """SimCSE NLI dataset for natural language inference."""

    language = "en"

    hf_name = "sentence-transformers/simcse-nli"
    split = "train"
    has_multiple_datasets = False
    query_name = "premise"
    positive_name = "hypothesis"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    preprocessor = nli_preprocessor
    loader = from_one_hf_dataset
