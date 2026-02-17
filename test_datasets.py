import datasets
from datasets import load_dataset
from typing import Set

# lang = "bn"  # or any of the 16 languages
# miracl = datasets.load_dataset(
#     "miracl/miracl", lang, trust_remote_code=True, split="train"
# )
# miracl["query"][0]
# miracl["positive_passages"][0]
# miracl["negative_passages"][0]


dataset = load_dataset("BeIR/arguana-generated-queries", split="train")


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
    mteb_query_texts: Set[str],
    mteb_corpus_texts: Set[str],
):
    """
    Filter BeIR/arguana-generated-queries dataset to remove examples
    that overlap with mteb/arguana evaluation set.
    """

    def is_not_overlapping(example):
        query_norm = normalize_text(example["query"])
        positive_norm = normalize_text(example["text"])

        # Remove if query OR positive text appears in MTEB evaluation set
        query_overlaps = query_norm in mteb_query_texts
        positive_overlaps = positive_norm in mteb_corpus_texts

        return not (query_overlaps or positive_overlaps)

    filtered_dataset = dataset.filter(is_not_overlapping)
    return filtered_dataset


mteb_query_texts, mteb_corpus_texts = get_mteb_arguana_texts()


original_size = len(dataset)

len(set(dataset["_id"]))

queries = load_dataset("mteb/arguana", name="queries", split="queries")


queries

dataset["query"]

dataset = clear_arguana_overlap(
    dataset,
    mteb_query_texts,
    mteb_corpus_texts,
)
filtered_size = len(dataset)

filtered_size
