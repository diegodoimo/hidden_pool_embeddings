from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
from typing import Set
from tasks.data_helpers import RetrievalRawData


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def get_mteb_arguana_texts() -> tuple[Set[str], Set[str]]:
    """
    Load mteb/arguana evaluation dataset and return sets of normalized
    query texts and corpus texts for deduplication.
    """
    # Load MTEB arguana corpus and queries
    corpus = load_dataset("mteb/arguana", name="corpus", split="corpus")
    queries = load_dataset("mteb/arguana", name="queries", split="queries")

    # Build sets of normalized texts
    corpus_texts = {normalize_text(row["text"]) for row in corpus}
    query_texts = {normalize_text(row["text"]) for row in queries}

    return query_texts, corpus_texts


def clear_arguana_overlap(
    dataset,
    anchor_field: str,
    positive_field: str,
    mteb_query_texts: Set[str],
    mteb_corpus_texts: Set[str],
):
    """
    Filter BeIR/arguana-generated-queries dataset to remove examples
    that overlap with mteb/arguana evaluation set.
    """

    def is_not_overlapping(example):
        query_norm = normalize_text(example[anchor_field])
        positive_norm = normalize_text(example[positive_field])

        # Remove if query OR positive text appears in MTEB evaluation set
        query_overlaps = query_norm in mteb_query_texts
        positive_overlaps = positive_norm in mteb_corpus_texts

        return not (query_overlaps or positive_overlaps)

    filtered_dataset = dataset.filter(is_not_overlapping)
    return filtered_dataset


def load_arguana_dedup_retrieval(task) -> RetrievalRawData:
    """Load BeIR/arguana-generated-queries with deduplication against mteb/arguana.

    This removes any query-positive pairs where either the query or positive text
    appears in the mteb/arguana evaluation set, preventing train-test contamination.
    """
    # Load the BeIR arguana dataset
    dataset = load_dataset(task.hf_name, split=task.split)

    # Get MTEB arguana texts for deduplication
    print("Loading mteb/arguana for deduplication...")
    mteb_query_texts, mteb_corpus_texts = get_mteb_arguana_texts()
    print(
        f"Found {len(mteb_query_texts)} MTEB queries and {len(mteb_corpus_texts)} MTEB corpus texts"
    )

    # Filter out overlapping examples
    original_size = len(dataset)
    dataset = clear_arguana_overlap(
        dataset,
        task.anchor_name,
        task.positive_name,
        mteb_query_texts,
        mteb_corpus_texts,
    )
    filtered_size = len(dataset)
    print(
        f"Removed {original_size - filtered_size} overlapping examples ({original_size} -> {filtered_size})"
    )

    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])
    document_texts = positive_texts.copy()

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        positive_titles=None,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=query_texts,
        unique_query_ids=query_ids,
        unique_positive_texts=positive_texts,
        unique_positive_ids=positive_ids,
        unique_positive_titles=None,
        corpus_dict=corpus_dict,
        has_title=False,
        documents_are_positives=True,
    )


class Arguana(AbsTask):
    """BeIR Arguana dataset with deduplication against mteb/arguana eval set."""

    language = "en"

    hf_name = "BeIR/arguana-generated-queries"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    negative_name = "negative"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["ArguAna"]})
    loader = load_arguana_dedup_retrieval
