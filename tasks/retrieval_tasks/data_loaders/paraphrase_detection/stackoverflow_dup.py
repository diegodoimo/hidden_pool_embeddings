from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import Dataset
from tasks.retrieval_loaders import from_one_hf_dataset


def stackoverflow_preprocessor(dataset, query_name, positive_name):
    """Flatten StackOverflow duplicate questions dataset.

    Extracts the first positive from the positive list for each query.
    Keeps the negative column as-is for corpus expansion.
    """
    queries = []
    positives = []
    negatives = []

    for row in dataset:
        query = row.get(query_name, "")
        pos_list = row.get(positive_name, [])
        neg_list = row.get("negative", [])

        if isinstance(pos_list, list) and len(pos_list) > 0:
            first_positive = pos_list[0]
        elif isinstance(pos_list, str) and pos_list:
            first_positive = pos_list
        else:
            continue

        if query:
            queries.append(query)
            positives.append(first_positive)
            if isinstance(neg_list, list):
                negatives.append(neg_list)
            elif neg_list:
                negatives.append([neg_list])
            else:
                negatives.append([])

    return Dataset.from_dict(
        {query_name: queries, positive_name: positives, "negative": negatives}
    )


class StackOverflowDupQuestions(AbsTask):
    """StackOverflow duplicate questions reranking dataset."""

    language = "en"

    hf_name = "mteb/stackoverflowdupquestions-reranking"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["StackOverflowDupQuestions"]}
    )
    loader = from_one_hf_dataset
    preprocessor = stackoverflow_preprocessor
