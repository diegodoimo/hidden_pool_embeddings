from datasets import load_dataset
from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value


def get_dict(dataset, id_field, text_field, title_field=None):

    ids = dataset[id_field]
    texts = dataset[text_field]

    if title_field:
        titles = dataset[title_field]
        return dict(zip(ids, zip(texts, titles)))
    else:
        return dict(zip(ids, texts))


def load_data_retrieval(task) -> Dataset:
    """
    Load retrieval task data and return as HuggingFace Dataset.

    Returns:
        Dataset with structure:
        {
            'queries': {'text': str, 'id': str},
            'positives': {'text': str, 'id': str, 'title': str (optional)},
            'documents': {'text': str, 'id': str, 'title': str (optional)}
        }
    """

    if task.has_multiple_datasets:

        print("Loading datasets...")
        qrels = load_dataset(task.hf_name, name=task.qrels_name, split=task.split)
        anchors_ = load_dataset(task.hf_name, name=task.anchor_name, split=task.anchor_name)
        corpus = load_dataset(task.hf_name, name=task.positive_name, split=task.positive_name)

        print("Mapping datasets to dict...")
        queries_dict = get_dict(anchors_, task.anchor_fields["id"], task.anchor_fields["text"])
        corpus_dict = get_dict(
            corpus,
            task.corpus_fields["id"],
            task.corpus_fields["text"],
            task.corpus_fields.get("title", None),
        )

        has_title = task.corpus_fields.get("title", None) is not None

        print("Extracting positives from qrels...")
        query_ids = []
        query_texts = []
        positive_ids = []
        positive_texts = []
        positive_titles = [] if has_title else None

        for qrel in qrels:
            anchor_id = qrel[task.qrels_fields["anchor_id"]]
            positive_id = qrel[task.qrels_fields["positive_id"]]
            score = qrel[task.qrels_fields["score"]]

            # Filter invalid pairs
            if anchor_id not in queries_dict or positive_id not in corpus_dict or score < 1:
                continue

            # Extract query
            query_text = queries_dict[anchor_id]
            query_ids.append(anchor_id)
            query_texts.append(query_text)

            # Extract positive
            positive_entry = corpus_dict[positive_id]
            positive_ids.append(positive_id)

            if has_title:
                text, title = positive_entry
                positive_texts.append(text)
                positive_titles.append(title)
            else:
                positive_texts.append(positive_entry)

        # Extract all documents from corpus
        document_ids = list(corpus_dict.keys())
        if has_title:
            document_texts = [text for text, title in corpus_dict.values()]
            document_titles = [title for text, title in corpus_dict.values()]
        else:
            document_texts = list(corpus_dict.values())
            document_titles = None

    else:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)

        # Assume dataset has matching lengths and indices correspond to pairs
        query_texts = list(dataset[task.anchor_name])
        positive_texts = list(dataset[task.positive_name])
        document_texts = list(dataset[task.positive_name])

        # Generate sequential IDs
        n_pairs = len(query_texts)
        query_ids = [f"query_{i}" for i in range(n_pairs)]
        positive_ids = [f"doc_{i}" for i in range(n_pairs)]

        # Documents use same IDs as positives
        document_ids = positive_ids.copy()

        # Check if titles exist in dataset
        has_title = (
            hasattr(task, "corpus_fields") and task.corpus_fields.get("title", None) is not None
        )
        if has_title and task.corpus_fields["title"] in dataset.column_names:
            positive_titles = list(dataset[task.corpus_fields["title"]])
            document_titles = positive_titles.copy()
        else:
            has_title = False
            positive_titles = None
            document_titles = None

    # Create HuggingFace Dataset
    hf_dataset = create_hf_dataset(
        query_texts,
        query_ids,
        positive_texts,
        positive_ids,
        positive_titles,
        document_texts,
        document_ids,
        document_titles,
        has_title,
    )

    return hf_dataset


def create_hf_dataset(
    query_texts,
    query_ids,
    positive_texts,
    positive_ids,
    positive_titles,
    document_texts,
    document_ids,
    document_titles,
    has_title,
):

    if has_title:
        positives_ = [
            {"text": text, "id": id_, "title": title}
            for text, id_, title in zip(positive_texts, positive_ids, positive_titles)
        ]
        documents_ = [
            {"text": text, "id": id_, "title": title}
            for text, id_, title in zip(document_texts, document_ids, document_titles)
        ]
    else:
        positives_ = [{"text": text, "id": id_} for text, id_ in zip(positive_texts, positive_ids)]
        documents_ = [{"text": text, "id": id_} for text, id_, in zip(document_texts, document_ids)]

    data = {
        "queries": [{"text": text, "id": id_} for text, id_ in zip(query_texts, query_ids)],
        "positives": positives_,
        "corpus": documents_,
    }

    # Define schema based on whether titles are present
    if has_title:
        features = Features(
            {
                "queries": {"text": Value("string"), "id": Value("string")},
                "positives": {
                    "text": Value("string"),
                    "id": Value("string"),
                    "title": Value("string"),
                },
                "corpus": {
                    "text": Value("string"),
                    "id": Value("string"),
                    "title": Value("string"),
                },
            }
        )
    else:
        features = Features(
            {
                "queries": {"text": Value("string"), "id": Value("string")},
                "positives": {"text": Value("string"), "id": Value("string")},
                "corpus": {"text": Value("string"), "id": Value("string")},
            }
        )
    hf_dataset = Dataset.from_dict(data, features=features)
    return hf_dataset


def load_data_classification(task, balance_dataset=True):

    dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)

    anchors = dataset[task.ancor_name]
    labels = dataset[task.label_name]

    return anchors, labels
