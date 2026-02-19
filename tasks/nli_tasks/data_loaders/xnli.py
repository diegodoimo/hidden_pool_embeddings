from tasks.abs_task import AbsTask, TaskMetadata
from tasks.nli_tasks.nli_loaders import nli_preprocessor
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


XNLI_LANGUAGES = [
    "ar",
    "bg",
    "de",
    "el",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]


class XNLI(AbsTask):
    """XNLI multilingual natural language inference dataset.

    Each language is loaded as a separate subtask.
    Set hf_subset to a specific language code to load only that language,
    or leave as None to load all languages via subtasks.
    """

    language = "multilingual"

    hf_name = "mteb/xnli"
    hf_subset = None
    split = "train"
    has_multiple_datasets = False
    query_name = "premise"
    positive_name = "hypothesis"
    negative_name = "negative"
    subtasks = XNLI_LANGUAGES
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    preprocessor = nli_preprocessor
    loader = from_one_hf_dataset
