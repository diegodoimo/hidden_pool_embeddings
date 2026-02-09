from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_cnndm_retrieval(task) -> RetrievalRawData:
    """Load CNN/DailyMail summarization dataset for retrieval.

    Highlights (summary) is used as query, article is used as positive.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset["highlights"])
    positive_texts = list(dataset["article"])
    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
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
    )


class CNNDM(AbsTask):
    """CNN/DailyMail summarization dataset for retrieval (highlights -> article)."""

    hf_name = "abisee/cnn_dailymail"
    hf_subset = "3.0.0"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "highlights"
    positive_name = "article"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original article"},
    )
    loader = load_cnndm_retrieval
