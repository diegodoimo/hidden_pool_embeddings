from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset
from tasks.data_helpers import RetrievalRawData


def load_pubmedqa_retrieval(task) -> RetrievalRawData:
    """Load PubMedQA dataset for retrieval.

    Question is query, context (long_answer) is positive document.
    """
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    query_texts = []
    positive_texts = []

    for row in dataset:
        question = row.get("question", "")
        # Context can be a list or dict with 'contexts' field
        context = row.get("context", {})
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            if contexts:
                positive_text = " ".join(contexts)
            else:
                continue
        elif isinstance(context, list):
            positive_text = " ".join(context)
        elif isinstance(context, str):
            positive_text = context
        else:
            continue

        if question and positive_text:
            query_texts.append(question)
            positive_texts.append(positive_text)

    document_texts = positive_texts.copy()

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


class PubMedQA(AbsTask):
    """PubMedQA biomedical QA dataset for retrieval."""

    language = "en"

    hf_name = "qiaojin/PubMedQA"
    hf_subset = "pqa_labeled"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )
    loader = load_pubmedqa_retrieval
