from datasets import load_dataset
import time
import torch.distributed as dist


def from_one_hf_dataset(task)::
    """
    Load data from a single HuggingFace dataset where queries and positives
    are in the same dataset with matching indices.

    Used by: NaturalQuestions, ALL_NLI, PAQ, ELI5, TriviaQA, COLIEE,
             S2ORC*, SPECTER, SentenceCompression, StackExchangeDup*, QQP, AmazonQA
    """
    rank = dist.get_rank()

    if rank == 0:
        start = time.time()
        print("Loading datasets...")
    if task.hf_subset:
        dataset = load_dataset(task.hf_name, name=task.hf_subset, split=task.split)
    else:
        dataset = load_dataset(task.hf_name, split=task.split)

    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60}min")
        start = time.time()
        print(f"building lists and finding unique items...")

    # Convert to list only once per column
    query_texts = list(dataset[task.anchor_name])
    positive_texts = list(dataset[task.positive_name])
    # Documents are the same as positives in this format - no need to convert again
    document_texts = positive_texts

    # Generate sequential IDs
    n_pairs = len(query_texts)
    query_ids = [f"query_{i}" for i in range(n_pairs)]
    positive_ids = [f"doc_{i}" for i in range(n_pairs)]
    # Documents use same IDs as positives - no need to copy
    document_ids = positive_ids

    # Check if titles exist in dataset
    has_corpus_fields = task.corpus_fields is not None
    has_title = has_corpus_fields and task.corpus_fields.get("title", None) is not None
    if has_title and task.corpus_fields["title"] in dataset.column_names:
        positive_titles = list(dataset[task.corpus_fields["title"]])
        document_titles = positive_titles  # Same as positives - no copy needed
    else:
        has_title = False
        positive_titles = None
        document_titles = None

    # Find unique queries and positives (similar to from_multiple_hf_datasets)
    seen_queries = set()
    seen_positives = set()
    unique_query_ids = []
    unique_query_texts = []
    unique_positive_ids = []
    unique_positive_texts = []
    unique_positive_titles = [] if has_title else None

    for i in range(n_pairs):
        # Track unique queries
        if query_texts[i] not in seen_queries:
            seen_queries.add(query_texts[i])
            unique_query_ids.append(query_ids[i])
            unique_query_texts.append(query_texts[i])

        # Track unique positives
        if positive_texts[i] not in seen_positives:
            seen_positives.add(positive_texts[i])
            unique_positive_ids.append(positive_ids[i])
            unique_positive_texts.append(positive_texts[i])
            if has_title:
                unique_positive_titles.append(positive_titles[i])

    if rank == 0:
        print(f"Found {len(unique_query_texts)} unique queries out of {n_pairs}")
        print(f"Found {len(unique_positive_texts)} unique positives out of {n_pairs}")
        print(f"lists built in {(time.time()-start)/60}min")
        start = time.time()
        print(f"generating corpus dict...")

    # Build corpus dict (using document_texts which is same as positive_texts)
    corpus_dict = {
        id_: {"text": doc_text} for id_, doc_text in zip(document_ids, document_texts)
    }

    if rank == 0:
        print(f"corpus dict built in {(time.time()-start)/60}min")

    return RetrievalRawData(
        query_texts=query_texts,
        query_ids=query_ids,
        positive_texts=positive_texts,
        positive_ids=positive_ids,
        positive_titles=positive_titles,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        unique_positive_texts=unique_positive_texts,
        unique_positive_ids=unique_positive_ids,
        unique_positive_titles=unique_positive_titles,
        corpus_dict=corpus_dict,
        has_title=has_title,
    )
