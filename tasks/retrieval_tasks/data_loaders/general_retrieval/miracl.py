from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


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
    "yo",
    "de",
]


def extract_unique_queries(
    query_texts: list[str],
    query_ids: list[str],
    positive_texts: list[str],
    positive_ids: list[str],
    positive_titles: list[str] | None = None,
) -> tuple:
    """Extract unique queries from lists that may contain repeated queries."""
    seen_query_texts = set()
    unique_query_texts = []
    unique_query_ids = []
    unique_positive_texts = []
    unique_positive_ids = []
    unique_positive_titles = [] if positive_titles is not None else None

    for i, query_text in enumerate(query_texts):
        if query_text not in seen_query_texts:
            seen_query_texts.add(query_text)
            unique_query_texts.append(query_text)
            unique_query_ids.append(query_ids[i])
            unique_positive_texts.append(positive_texts[i])
            unique_positive_ids.append(positive_ids[i])
            if positive_titles is not None:
                unique_positive_titles.append(positive_titles[i])

    return (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    )


def load_miracl_retrieval(task) -> RetrievalRawData:
    """Load MIRACL multilingual retrieval dataset.

    If hf_subset is None, loads all available languages.
    """
    query_texts = []
    positive_texts = []

    if task.hf_subset:
        # Load single language
        languages = [task.hf_subset]
    else:
        # Load all languages
        languages = MIRACL_LANGUAGES
        print(f"Loading MIRACL for all {len(languages)} languages...")

    for lang in languages:
        try:
            dataset = load_dataset(task.hf_name, name=lang, split=task.split)
            for row in dataset:
                query = row["query"]
                # positive_passages is a list of dicts with 'text' field
                positives = row.get("positive_passages", [])
                if positives and len(positives) > 0:
                    query_texts.append(query)
                    # Take the first positive passage
                    positive_texts.append(positives[0].get("text", ""))
            if not task.hf_subset:
                print(f"  Loaded {lang}: {len(dataset)} samples")
        except Exception as e:
            print(f"  Skipping {lang}: {e}")
            continue

    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    # Extract unique queries
    (
        unique_query_texts,
        unique_query_ids,
        unique_positive_texts,
        unique_positive_ids,
        unique_positive_titles,
    ) = extract_unique_queries(
        query_texts, query_ids, positive_texts, positive_ids, None
    )

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=False,
        documents_are_positives=True,
    )


class MIRACL(AbsTask):
    """MIRACL multilingual retrieval dataset.

    Note: Each language is a separate config. Set hf_subset to specific language
    (ar, bn, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, zh, yo, de)
    or None to load all languages (requires custom loader modification).
    """

    language = "multilingual"

    hf_name = "miracl/miracl"
    hf_subset = None  # Load all available languages
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "positive_passages"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )
    loader = load_miracl_retrieval
