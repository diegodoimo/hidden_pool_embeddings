from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
from tasks.retrieval_loaders import from_one_hf_dataset


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def clear_arguana_overlap_mteb(
    dataset,
    query_field: str,
    positive_field: str,
):
    """
    Filter BeIR/arguana-generated-queries dataset to remove examples
    that overlap with mteb/arguana evaluation set.

    Loads the MTEB arguana corpus, queries and qrels internally to build
    the reference sets used for filtering.
    """

    corpus = load_dataset("mteb/arguana", name="corpus", split="corpus")
    corpus_dict = dict(zip(corpus["_id"], corpus["text"]))
    queries = load_dataset("mteb/arguana", name="queries", split="queries")
    qrels = load_dataset("mteb/arguana", name="default", split="test")
    positive_ids = set(qrels["corpus-id"])

    mteb_corpus_texts = {
        normalize_text(corpus_dict[id_]) for id_ in positive_ids if id_ in corpus_dict
    }

    # Build sets of normalized texts
    mteb_query_texts = {normalize_text(row["text"]) for row in queries}

    def is_not_overlapping(example):
        query_norm = normalize_text(example[query_field])
        positive_norm = normalize_text(example[positive_field])

        # Remove if query OR positive text appears in MTEB evaluation set
        query_overlaps = query_norm in mteb_query_texts
        positive_overlaps = positive_norm in mteb_corpus_texts

        return not (query_overlaps or positive_overlaps)

    filtered_dataset = dataset.filter(is_not_overlapping)
    return filtered_dataset


class Arguana(AbsTask):
    """BeIR Arguana dataset with deduplication against mteb/arguana eval set."""

    language = "en"

    hf_name = "BeIR/arguana-generated-queries"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    title_name = "title"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["ArguAna"]})
    loader = from_one_hf_dataset
    decontaminator = clear_arguana_overlap_mteb
