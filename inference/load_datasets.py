from datasets import load_dataset
from tasks.retrieval_tasks import *
from datasets import Dataset, Features, Value
import time 
import os

RANK = int(os.environ["RANK"])
# def get_dict(dataset, id_field, text_field, title_field=None):

#     ids = dataset[id_field]
#     texts = dataset[text_field]

#     if title_field:
#         titles = dataset[title_field]
#         return dict(zip(ids, zip(texts, titles)))
#     else:
#         return dict(zip(ids, texts))



# def get_dict(dataset, id_field, text_field, title_field=None):

#     if title_field:
#         return {row[id_field]: (row[text_field], row[title_field]) 
#                 for row in dataset}
#     else:
#         return {row[id_field]: row[text_field] 
#                 for row in dataset}


# def get_dict(dataset, id_field, text_field, title_field=None):
#     # Access columns directly instead of iterating rows
#     ids = dataset[id_field]
#     texts = dataset[text_field]
    
#     if title_field:
#         titles = dataset[title_field]
#         return {id_: (text, title) 
#                 for id_, text, title in zip(ids, texts, titles)}
#     else:
#         return {id_: text for id_, text in zip(ids, texts)}




from multiprocessing import Pool
import os

def process_chunk(args):
    chunk, id_field, text_field, title_field = args
    if title_field:
        return {row[id_field]: (row[text_field], row[title_field]) 
                for row in chunk}
    else:
        return {row[id_field]: row[text_field] for row in chunk}

def get_dict(dataset, id_field, text_field, title_field=None):
    #n_workers = os.cpu_count()-2
    n_workers = 16
    chunk_size = len(dataset) // n_workers
    
    chunks = [dataset.select(range(i, min(i + chunk_size, len(dataset)))) 
              for i in range(0, len(dataset), chunk_size)]
    
    with Pool(n_workers) as pool:
        results = pool.map(process_chunk, 
                          [(chunk, id_field, text_field, title_field) 
                           for chunk in chunks])
    
    # Merge dictionaries
    return {k: v for d in results for k, v in d.items()}



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

        
        if RANK ==0:
            print(f"Mapping {len(anchors_)} queries to dict...")
            start = time.time()
        queries_dict = get_dict(anchors_, task.anchor_fields["id"], task.anchor_fields["text"])

        if RANK ==0: 
            print(f"{(time.time() - start): .2f} sec for {len(queries_dict)} samples")
            start = time.time()
            print(f"Mapping {len(corpus)} docs to dict...")
        corpus_dict = get_dict(
            corpus,
            task.corpus_fields["id"],
            task.corpus_fields["text"],
            task.corpus_fields.get("title", None),
        )

        has_title = task.corpus_fields.get("title", None) is not None
        if RANK ==0: 
            print(f"{(time.time() - start)/60: .2f} min for {len(corpus_dict)} samples")
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
            query_ids.append(anchor_id)
            query_texts.append(queries_dict[anchor_id])

            # Extract positive
            positive_entry = corpus_dict[positive_id]
            positive_ids.append(positive_id)

            if has_title:
                text, title = positive_entry
                positive_texts.append(text)
                positive_titles.append(title)
            else:
                positive_texts.append(positive_entry)

        # queries can be repeted many times in the search
        # for negatives we just want unique queries
        unique_query_ids = set(query_ids)
        unique_queries = [queries_dict[id_] for id_ in unique_query_ids]

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
        unique_queries,
        unique_query_ids,
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
    unique_queries,
    unique_ids,
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

    queries_ds = Dataset.from_dict(
        {
            "text": query_texts,
            "id": query_ids,
        },
        features=Features(
            {
                "text": Value("string"),
                "id": Value("string"),
            }
        ),
    )

    unique_queries_ds = Dataset.from_dict(
        {
            "text": unique_queries,
            "id": unique_ids,
        },
        features=Features(
            {
                "text": Value("string"),
                "id": Value("string"),
            }
        ),
    )

    if has_title:

        positives_ds = Dataset.from_dict(
            {
                "text": positive_texts,
                "id": positive_ids,
                "title": positive_titles,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                    "title": Value("string"),
                }
            ),
        )

        corpus_ds = Dataset.from_dict(
            {
                "text": document_texts,
                "id": document_ids,
                "title": document_titles,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                    "title": Value("string"),
                }
            ),
        )
    else:

        positives_ds = Dataset.from_dict(
            {
                "text": positive_texts,
                "id": positive_ids,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                }
            ),
        )

        corpus_ds = Dataset.from_dict(
            {
                "text": document_texts,
                "id": document_ids,
            },
            features=Features(
                {
                    "text": Value("string"),
                    "id": Value("string"),
                }
            ),
        )

    return {
        "unique_queries": unique_queries_ds,
        "queries": queries_ds,
        "positives": positives_ds,
        "corpus": corpus_ds,
    }


def load_data_classification(task, balance_dataset=True):

    dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)

    anchors = dataset[task.ancor_name]
    labels = dataset[task.label_name]

    return anchors, labels
