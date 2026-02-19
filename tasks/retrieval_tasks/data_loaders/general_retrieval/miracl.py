from tasks.abs_task import AbsTask, TaskMetadata
from datasets import Dataset
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


# yo (Yoruba) and de (German) are MIRACL+ additions with no train split.
MIRACL_LANGUAGES = [
    "ar",
    "bn",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "sw",
    "te",
    "th",
    "zh",
]


def miracl_preprocessor(dataset, query_name, positive_name):
    """Flatten MIRACL dataset: explode positive_passages list into one row per pair.

    Each positive passage dict (with docid, text, title) becomes a separate row.
    Negative passages are kept as parallel lists of texts and titles for corpus
    expansion via from_one_hf_dataset's negative handling.
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

        # Extract negative texts and titles (same for all positives of this query)
        neg_texts = [p.get("text", "") for p in neg_passages if p.get("text")]
        neg_t = [p.get("title", "") for p in neg_passages if p.get("text")]

        # Each positive becomes a separate row
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


class MIRACL(AbsTask):
    """MIRACL multilingual retrieval dataset.

    Each language is loaded as a separate subtask.
    Set hf_subset to a specific language code to load only that language,
    or leave as None to load all languages via subtasks.

    Uses only the train split to avoid contamination with MTEB evaluation:
    MIRACLRetrievalHardNegatives is evaluated on the dev split, which
    corresponds to the dev data of the original miracl/miracl dataset.
    """

    language = "multilingual"

    hf_name = "miracl/miracl"
    hf_subset = None
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "positive_passages"
    negative_name = "negative"
    corpus_fields = {"title": "title"}
    subtasks = MIRACL_LANGUAGES
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    trust_remote_code = True
    loader = from_one_hf_dataset
    preprocessor = miracl_preprocessor
