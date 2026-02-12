from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_stackexchange_retrieval(task) -> RetrievalRawData:
    """Load StackExchange dataset for retrieval.

    Title+body is used as query, upvoted_answer is used as positive document.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    # Combine title and body for query
    query_texts = [f"{row['title']} {row['body']}" for row in dataset]
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


class StackExchangeRetrieval(AbsTask):
    """StackExchange dataset for retrieval - title+body as query, answer as positive."""

    language = "en"

    hf_name = "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
    hf_subset = "apple"  # Default subset, can be changed
    split = "train"
    has_multiple_datasets = False
    anchor_name = "title_body"
    positive_name = "upvoted_answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve answers that best answer the question"
        },
    )
    loader = load_stackexchange_retrieval
