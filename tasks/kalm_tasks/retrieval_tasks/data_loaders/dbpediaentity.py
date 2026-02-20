from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset


class DBpediaEntity(AbsTask):
    """DBpedia-Entity dataset for entity retrieval."""

    language = "en"

    hf_name = "sentence-transformers/dbpedia-entity"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions"
        },
    )
    loader = from_one_hf_dataset
