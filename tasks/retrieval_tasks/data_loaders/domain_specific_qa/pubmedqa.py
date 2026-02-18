from tasks.abs_task import AbsTask, TaskMetadata
from datasets import Dataset
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


PUBMEDQA_SUBTASKS = ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]


def pubmedqa_preprocessor(dataset, query_name: str, positive_name: str) -> Dataset:
    """Flatten PubMedQA's nested context field into a plain string column.

    PubMedQA stores context as a dict with a 'contexts' key (list of strings).
    This preprocessor joins those strings and returns a HuggingFace Dataset
    with flat `query_name` and `positive_name` columns, dropping rows where
    either field is empty.
    """
    query_texts = []
    positive_texts = []

    for row in dataset:
        question = row.get("question", "")
        context = row.get("context", {})
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if contexts:
                positive_text = " ".join(contexts)
            else:
                continue
        elif isinstance(context, list):
            positive_text = " ".join(context)
        elif isinstance(context, str):
            positive_text = context
        else:
            continue

        if question and positive_text:
            query_texts.append(question)
            positive_texts.append(positive_text)

    return Dataset.from_dict({query_name: query_texts, positive_name: positive_texts})


class PubMedQA(AbsTask):
    """PubMedQA biomedical QA dataset for retrieval."""

    language = "en"

    hf_name = "qiaojin/PubMedQA"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
    preprocessor = pubmedqa_preprocessor
    subtasks = PUBMEDQA_SUBTASKS
