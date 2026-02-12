from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_squad_retrieval(task) -> RetrievalRawData:
    """Load SQuAD dataset for retrieval.

    Question is used as query, context is used as positive document.
    """
    dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])

    # Remove duplicate contexts to create corpus
    seen_contexts = {}
    document_texts = []
    document_ids = []
    positive_ids = []

    for i, context in enumerate(positive_texts):
        if context not in seen_contexts:
            doc_id = f"doc_{len(seen_contexts)}"
            seen_contexts[context] = doc_id
            document_texts.append(context)
            document_ids.append(doc_id)
        positive_ids.append(seen_contexts[context])

    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]

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


class SQuAD(AbsTask):
    """SQuAD dataset for retrieval - question as query, context as positive."""

    language = "en"

    hf_name = "rajpurkar/squad"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )
    loader = load_squad_retrieval
