from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_xsum_retrieval(task) -> RetrievalRawData:
    """Load XSum summarization dataset for retrieval.

    Summary is used as query, document is used as positive.
    """
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset["summary"])
    positive_texts = list(dataset["document"])
    document_texts = positive_texts.copy()

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    document_ids = positive_ids.copy()

    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }
    
    # All documents are positives in this dataset
    n_positives = len(document_ids)

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=None,
        unique_query_texts=query_texts,
        unique_query_ids=query_ids,
        corpus_dict=corpus_dict,
        has_title=False,
        n_positives=n_positives,
    )


class XSum(AbsTask):
    """XSum summarization dataset for retrieval (summary -> document)."""

    language = "en"

    hf_name = "EdinburghNLP/xsum"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "summary"
    positive_name = "document"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original document"},
    )
    loader = load_xsum_retrieval
