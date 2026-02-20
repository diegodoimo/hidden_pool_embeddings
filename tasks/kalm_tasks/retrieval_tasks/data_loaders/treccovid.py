from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_loaders import from_one_hf_dataset
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def clear_treccovid_overlap_mteb(
    dataset,
    query_field: str,
    positive_field: str,
):
    """
    Filter sentence-transformers/trec-covid dataset to remove examples
    that overlap with the mteb/trec-covid evaluation set.

    Loads the MTEB TRECCOVID corpus, queries and qrels internally to build
    the reference sets used for filtering. MTEB evaluates TRECCOVID on the
    test split, which contains 50 COVID-19 queries with relevance judgments
    over the CORD-19 corpus.
    """

    corpus = load_dataset("mteb/trec-covid", name="corpus", split="corpus")
    corpus_dict = dict(zip(corpus["_id"], corpus["text"]))
    queries = load_dataset("mteb/trec-covid", name="queries", split="queries")
    qrels = load_dataset("mteb/trec-covid", name="default", split="test")
    positive_ids = set(qrels["corpus-id"])

    mteb_corpus_texts = {
        normalize_text(corpus_dict[id_]) for id_ in positive_ids if id_ in corpus_dict
    }
    mteb_query_texts = {normalize_text(row["text"]) for row in queries}

    def is_not_overlapping(example):
        query_norm = normalize_text(example[query_field])
        positive_norm = normalize_text(example[positive_field])
        return not (query_norm in mteb_query_texts or positive_norm in mteb_corpus_texts)

    return dataset.filter(is_not_overlapping)


class TRECCOVID(AbsTask):
    """TREC-COVID dataset for COVID-19 scientific article retrieval.

    Uses a decontaminator to remove overlap with the MTEB TRECCOVID evaluation
    set (mteb/trec-covid, test split), which appears in both the MTEB English
    and MTEB Multilingual benchmarks.
    """

    language = "en"

    hf_name = "sentence-transformers/trec-covid"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "query"
    positive_name = "text"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a query about COVID-19, retrieve relevant scientific passages"
        },
    )
    loader = from_one_hf_dataset
    decontaminator = clear_treccovid_overlap_mteb
