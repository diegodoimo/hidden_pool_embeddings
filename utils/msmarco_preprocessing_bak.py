
def prepare_msmarco(hf_queries, hf_corpus, hf_qrels):
    query_to_doc_map = {
        example["query-id"]: example["corpus-id"] for example in hf_qrels if example["score"] == 1
    }

    # Get only the IDs we need
    relevant_doc_ids = set(query_to_doc_map.values())
    relevant_query_token_ids = set(query_to_doc_map.keys())

    # Filter while loading (still batched for speed)
    def build_filtered_dict(dataset, relevant_ids, batch_size=10000):
        filtered_dict = {}
        for batch in dataset.iter(batch_size=batch_size):
            for i in range(len(batch["_id"])):
                doc_id = batch["_id"][i]
                if doc_id in relevant_ids:
                    filtered_dict[doc_id] = {
                        "_id": doc_id,
                        "title": batch["title"][i] if "title" in batch else None,
                        "text": batch["text"][i],
                    }
        return filtered_dict

    corpus_full = build_filtered_dict(hf_corpus["corpus"], relevant_doc_ids)
    queries_full = {item["_id"]: item for item in hf_queries["queries"]}

    # Build aligned data
    data_query = {"query_id": [], "query_text": [], "positive_id": []}

    data_doc = {"positive_id": [], "positive_title": [], "positive_text": []}

    for query_id, doc_id in query_to_doc_map.items():
        if query_id in queries_full and doc_id in corpus_full:
            data_query["query_id"].append(query_id)
            data_query["query_text"].append(queries_full[query_id]["text"])
            data_query["positive_id"].append(doc_id)

            data_doc["positive_id"].append(doc_id)
            title = corpus_full[doc_id]["title"]
            data_doc["positive_title"].append(title if len(title) > 0 else "none")
            data_doc["positive_text"].append(corpus_full[doc_id]["text"])

    # Create HuggingFace dataset
    # train_queries = Dataset.from_dict(data_query)
    # train_docs = Dataset.from_dict(data_doc)
    return data_query, data_doc




def tokenize_batch(
    examples,
    query_prompt,
    doc_prompt,
    tokenizer,
    max_query_len,
    max_passage_len,
    num_hard_negatives=None,
) -> Dict[str, List]:
    """Tokenize a batch of examples with prompts."""
    batch_size = len(examples["query"])

    # Prepend prompts
    queries_with_prompt = [query_prompt + q for q in examples["query"]]
    pos_with_prompt = [
        doc_prompt.format(title=title, text=text)
        for title, text in zip(examples["pos_title"], examples["pos_passage"])
    ]

    # Tokenize queries
    query_encs = tokenizer(
        queries_with_prompt,
        max_length=max_query_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    # Tokenize positive passages
    pos_encs = tokenizer(
        pos_with_prompt,
        max_length=max_passage_len,
        truncation=True,
        padding=False,
        return_attention_mask=False,
    )

    # Initialize output
    result = {
        "query_text": queries_with_prompt,
        "pos_text": pos_with_prompt,
        "query_token_ids": query_encs["input_ids"],
        "pos_token_ids": pos_encs["input_ids"],
        "pos_ids": [int(ids) for ids in examples["pos_ids"]],
        "query_len": [len(ids) for ids in query_encs["input_ids"]],
        "pos_len": [len(ids) for ids in pos_encs["input_ids"]],
        "total_len": [],
    }

    # Process negatives if available
    if "neg_passages" in examples:
        for i in range(batch_size):
            neg_passages = examples["neg_passages"][i]

            # Take first num_hard_negatives
            neg_passages_subset = neg_passages[:num_hard_negatives]

            # Prepend document prompt to negatives
            neg_with_prompt = [doc_prompt + neg for neg in neg_passages_subset]

            # Tokenize negatives
            neg_encs = tokenizer(
                neg_with_prompt,
                max_length=max_passage_len,
                truncation=True,
                padding=False,
                return_attention_mask=False,
            )

            neg_token_ids_list = neg_encs["input_ids"]

            # Pad with positive passage if not enough negatives
            # while len(neg_token_ids_list) < num_hard_negatives:
            #     neg_token_ids_list.append(pos_encs["input_ids"][i])

            result["neg_token_ids"].append(neg_token_ids_list)

            # Calculate average negative length
            avg_neg_len = np.mean([len(neg) for neg in neg_token_ids_list])
            result["avg_neg_len"].append(avg_neg_len)
            result["total_len"].append(
                result["query_len"][i] + result["pos_len"][i] + avg_neg_len * num_hard_negatives
            )
    else:
        # No negatives provided
        for i in range(batch_size):
            # result["neg_token_ids"].append([pos_encs["input_ids"][i]] * num_hard_negatives)
            # result["avg_neg_len"].append(result["pos_len"][i])
            # result["total_len"].append(
            #     result["query_len"][i] + result["pos_len"][i] * (1 + num_hard_negatives)
            # )
            result["total_len"].append(result["query_len"][i] + result["pos_len"][i])

    return result





def msmarco_dataset(
    queries_dataset: Dataset,
    pos_passages_dataset: Dataset,
    tokenizer,
    max_query_len: int = 32,
    max_passage_len: int = 256,
    num_hard_negatives: int = 7,
    sort_by_length: bool = True,
    neg_passages_dataset: Optional[Dataset] = None,
    query_task: str = "Retrieval-query",
    document_task: str = "Retrieval-document",
    batch_size: int = 1000,
    rank=None,
) -> Dataset:
    """
    Prepares MS MARCO dataset with batched processing for efficiency.

    Args:
        queries_dataset: HF Dataset with 'text' column
        pos_passages_dataset: HF Dataset with 'text' column
        tokenizer: HuggingFace tokenizer
        max_query_len: Maximum query length
        max_passage_len: Maximum passage length
        num_hard_negatives: Number of hard negatives per query
        sort_by_length: Whether to sort by total length
        neg_passages_dataset: Optional HF Dataset with 'text' column (list of negatives)
        query_task: Task type for query prompt
        document_task: Task type for document prompt
        batch_size: Batch size for processing

    Returns:
        HF Dataset with tokenized and optionally sorted data
    """

    assert len(queries_dataset) == len(pos_passages_dataset)
    if neg_passages_dataset is not None:
        assert len(queries_dataset) == len(neg_passages_dataset)

    # Get prompts
    query_prompt = TASK_PROMPTS[query_task]
    doc_prompt = TASK_PROMPTS[document_task]
    if rank is None or rank == 0:
        print(f"Query prompt: '{query_prompt}'")
        print(f"Document prompt: '{doc_prompt}'")

    # Combine datasets
    if neg_passages_dataset is not None:
        combined = Dataset.from_dict(
            {
                "query": queries_dataset["query_text"],
                "pos_passage": pos_passages_dataset["positive_text"],
                "pos_ids": pos_passages_dataset["positive_id"],
                "neg_passages": neg_passages_dataset["negative_text"],
                "neg_ids": neg_passages_dataset["nagative_id"],
            }
        )
    else:
        combined = Dataset.from_dict(
            {
                "query": queries_dataset["query_text"],
                "pos_passage": pos_passages_dataset["positive_text"],
                "pos_title": pos_passages_dataset["positive_title"],
                "pos_ids": pos_passages_dataset["positive_id"],
            }
        )

    if rank is None or rank == 0:
        print(f"Tokenizing {len(combined)} examples with batch_size={batch_size}...")

    tokenize_batch = partial(
        tokenize_batch,
        query_prompt=query_prompt,
        doc_prompt=doc_prompt,
        tokenizer=tokenizer,
        max_query_len=max_query_len,
        max_passage_len=max_passage_len,
        num_hard_negatives=num_hard_negatives,
    )

    # Apply batched tokenization
    tokenized_dataset = combined.map(
        tokenize_batch, batched=True, batch_size=batch_size, remove_columns=combined.column_names
    )

    tot_tokens = np.sum(tokenized_dataset["total_len"])
    # Sort by length if requested
    if sort_by_length:
        tokenized_dataset = tokenized_dataset.sort("total_len", reverse=True)

    if rank is None or rank == 0:
        print(f"{tot_tokens/10**6: .1f}M tokens")
        print(f"{len(tokenized_dataset)/10**3: .1f}k query-pas pairs")
        print(f"most long: {tokenized_dataset["total_len"][:30]}")
        print(f"avg query len: {np.mean(tokenized_dataset["query_len"])}")
        print(f"avg doc len: {np.mean(tokenized_dataset["pos_len"])}")

    return tokenized_dataset
