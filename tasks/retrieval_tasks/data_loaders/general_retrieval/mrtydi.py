from tasks.abs_task import AbsTask, TaskMetadata
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset
from datasets import Dataset


MRTYDI_SUBTASKS = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]


def mrtydi_preprocessor(dataset, query_name, positive_name):
    """Flatten Mr.TyDi dataset: explode positive_passages list into one row per pair.

    The schema mirrors MIRACL: each row has ``query``, ``positive_passages``
    (list of {docid, text, title}) and ``negative_passages`` (same).
    """
    queries = []
    positives = []
    titles = []
    negatives = []
    negative_titles = []

    for row in dataset:
        query = row.get(query_name, "")
        pos_passages = row.get(positive_name, [])
        neg_passages = row.get("negative_passages", [])

        if not query or not pos_passages:
            continue

        neg_texts = [p.get("text", "") for p in neg_passages if p.get("text")]
        neg_t = [p.get("title", "") for p in neg_passages if p.get("text")]

        for pos in pos_passages:
            pos_text = pos.get("text", "")
            if not pos_text:
                continue
            queries.append(query)
            positives.append(pos_text)
            titles.append(pos.get("title", ""))
            negatives.append(neg_texts)
            negative_titles.append(neg_t)

    return Dataset.from_dict(
        {
            query_name: queries,
            positive_name: positives,
            "title": titles,
            "negative": negatives,
            "negative_title": negative_titles,
        }
    )


class MrTyDi(AbsTask):
    """Mr.TyDi multilingual retrieval dataset (castorini/mr-tydi).

    Uses the original castorini dataset which has train splits with
    positive/negative passages per query (same format as MIRACL).
    mteb/mrtidy is evaluation-only (test split only) and cannot be
    used for training data collection.
    """

    language = "multilingual"

    hf_name = "castorini/mr-tydi"
    hf_subset = None
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "positive_passages"
    negative_name = "negative"
    corpus_fields = {"title": "title"}
    subtasks = MRTYDI_SUBTASKS
    trust_remote_code = True
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    loader = from_one_hf_dataset
    preprocessor = mrtydi_preprocessor
