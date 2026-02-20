from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_helpers import nli_preprocessor
from tasks.retrieval_loaders import from_one_hf_dataset


class ANLI(AbsTask):
    """ANLI dataset for retrieval - premise as query, entailed hypothesis as positive.

    By default, uses all three training rounds (train_r1, train_r2, train_r3) jointly.
    """

    language = "en"

    hf_name = "facebook/anli"
    split = "train_r1+train_r2+train_r3+dev_r1+dev_r2+dev_r3"
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
