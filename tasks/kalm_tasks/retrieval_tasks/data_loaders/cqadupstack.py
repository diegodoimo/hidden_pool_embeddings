from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset
from datasets import load_dataset


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and stripping whitespace."""
    return text.lower().strip()


def clear_cqadupstack_overlap_mteb(
    dataset,
    query_field: str,
    positive_field: str,
):
    """
    Filter sentence-transformers/cqadupstack dataset to remove examples
    that overlap with the MTEB CQADupstack evaluation sets.

    MTEB evaluates CQADupstackGamingRetrieval and CQADupstackUnixRetrieval
    on the test split (both appear in the MTEB English benchmark). The
    underlying question texts from these forums are shared between our
    training pairs and the MTEB retrieval queries, so we filter out any
    training pair whose question or duplicate text matches an MTEB test query.
    """

    mteb_query_texts = set()
    for forum in ("gaming", "unix"):
        queries = load_dataset(
            f"mteb/cqadupstack-{forum}", name="queries", split="queries"
        )
        mteb_query_texts.update(normalize_text(row["text"]) for row in queries)

    def is_not_overlapping(example):
        question_norm = normalize_text(example[query_field])
        duplicate_norm = normalize_text(example[positive_field])
        return not (
            question_norm in mteb_query_texts or duplicate_norm in mteb_query_texts
        )

    return dataset.filter(is_not_overlapping)


class CQADupStack(AbsTask):
    """CQADupStack dataset for duplicate question retrieval.

    Uses a decontaminator to remove overlap with the MTEB CQADupstack
    evaluation sets: CQADupstackGamingRetrieval and CQADupstackUnixRetrieval
    (both evaluated on test split in the MTEB English benchmark). Question
    texts from the Gaming and Unix StackExchange forums appear as retrieval
    queries in MTEB, so we filter any training pair containing those texts.
    """

    language = "en"

    hf_name = "sentence-transformers/cqadupstack"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    query_name = "question"
    positive_name = "duplicate"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve duplicate questions"
        },
    )
    loader = from_one_hf_dataset
    decontaminator = clear_cqadupstack_overlap_mteb
