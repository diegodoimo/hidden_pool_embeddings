from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


class NaturalQuestions(AbsTask):
    language = "en"
    hf_name = "sentence-transformers/natural-questions"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["NQ"]})
    loader = from_one_hf_dataset
