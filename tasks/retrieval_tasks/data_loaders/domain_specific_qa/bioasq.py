from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class BioASQ(AbsTask):
    """BioASQ biomedical QA dataset for retrieval."""

    language = "en"

    hf_name = "BeIR/bioasq-generated-queries"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    negative_name = "negative"
    corpus_fields = {"title": "title"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
