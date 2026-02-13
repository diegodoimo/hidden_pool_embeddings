from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_stackoverflow_dup_retrieval(task) -> RetrievalRawData:
    """Load StackOverflow duplicate questions dataset for retrieval."""
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = []
    positive_texts = []

    for row in dataset:
        query = row.get("query", "")
        # positive can be a list of positive examples
        positives = row.get("positive", [])
        if isinstance(positives, list) and len(positives) > 0:
            query_texts.append(query)
            positive_texts.append(positives[0])
        elif isinstance(positives, str) and positives:
            query_texts.append(query)
            positive_texts.append(positives)

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


class StackOverflowDupQuestions(AbsTask):
    """StackOverflow duplicate questions reranking dataset."""

    language = "en"

    hf_name = "mteb/stackoverflowdupquestions-reranking"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["StackOverflowDupQuestions"]}
    )
    loader = load_stackoverflow_dup_retrieval
