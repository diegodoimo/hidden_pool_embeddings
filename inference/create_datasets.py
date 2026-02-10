from mteb.types import PromptType
from functools import partial
from datasets import Dataset


from datasets import disable_progress_bars

disable_progress_bars()


# EMBEDDINGGEMMA
TASK_PROMPTS = {
    "document": "title: {title} | text: ",
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: ",
}


def instruction_template_qwen3(prompt_type, task_metadata, text, title="") -> str:
    # text = row["text"]

    if prompt_type == PromptType.query:
        if task_metadata.prompt is not None:
            instruction = task_metadata.prompt["query"]
            # just to mimick the broken mteb code
            prompt = f"Instruct: {instruction.strip()}\nQuery:{text.strip()}"
        else:
            prompt = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text.strip()}"

    elif prompt_type == PromptType.document:

        # title = row.get("title") or ""  # Use .get and default to empty string

        if len(title) > 0:
            prompt = f"{title} {text.strip()}"
        else:
            prompt = text.strip()  # Just return the text if no title

    return prompt


def _is_valid_corpus_row(row: dict[str, str]) -> bool:
    """Check if a corpus row has non-empty text content."""
    if "text" not in row or not row["text"] or not row["text"].strip():
        return False
    return True


def _is_valid_query_row(row: dict[str, str]) -> bool:
    """Check if a query row has non-empty text content."""
    if "text" not in row or not row["text"] or not row["text"].strip():
        return False
    return True


def _remove_long_sequences(rows, tokenizer, max_length):
    """Remove rows where the tokenized prompt exceeds max_length.

    Returns:
        tuple: (keep_mask, removed_ids)
            - keep_mask: list of booleans indicating which rows to keep
            - removed_ids: list of IDs that were removed
    """
    keep_mask = []
    removed_ids = []
    for i, prompt in enumerate(rows["prompt"]):
        # Fast path: if char length is very short, definitely keep
        if len(prompt) <= max_length:
            keep_mask.append(True)
        else:
            # Must tokenize to check actual token length
            token_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_length > max_length:
                keep_mask.append(False)
                removed_ids.append(rows["id"][i])
            else:
                keep_mask.append(True)
    return keep_mask, removed_ids


def _build_prompt(
    rows,
    tokenizer,
    instruction_template,
    prompt_type,
    task_metadata,
    eot_id,
):

    # at this stage we have {"id": [id1, id2, id3, ...], "text": [text1, text2, text3, ...], }
    num_rows = len(rows["text"])
    row_dicts = [{key: rows[key][i] for key in rows.keys()} for i in range(num_rows)]

    text_prompts = [
        instruction_template(prompt_type, task_metadata, row=row) for row in row_dicts
    ]
    # we use the dafault add_special_tokens = True, tokenizer.encode do not add the special token
    tokens = [tokenizer.encode(prompt) + [eot_id] for prompt in text_prompts]

    new_rows = {
        "id": rows["id"],
        "input_ids": tokens,
        "prompt": text_prompts,
        "text": rows["text"],
    }

    return new_rows


def _build_prompt(
    rows,
    tokenizer,
    instruction_template,
    prompt_type,
    task_metadata,
    eot_id,
):

    # at this stage we have {"id": [id1, id2, id3, ...], "text": [text1, text2, text3, ...], }
    # num_rows = len(rows["text"])
    # row_dicts = [{key: rows[key][i] for key in rows.keys()} for i in range(num_rows)]

    titles = rows.get("title", None)
    if titles:
        text_prompts = [
            instruction_template(prompt_type, task_metadata, text, title)
            for text, title in zip(rows["text"], rows["title"])
        ]
    else:
        text_prompts = [
            instruction_template(prompt_type, task_metadata, text)
            for text in rows["text"]
        ]
    # we use the dafault add_special_tokens = True, tokenizer.encode do not add the special token
    # tokens = [tokenizer.encode(prompt) + [eot_id] for prompt in text_prompts]

    # tokens = tokenizer(
    #     text_prompts,
    #     add_special_tokens=False,
    #     return_attention_mask=False,
    # )["input_ids"]

    # tokens = [tok + [eot_id] for tok in tokens]

    new_rows = {
        "id": rows["id"],
        "prompt": text_prompts,
        "text": rows["text"],
    }
    # "input_ids": tokens,
    return new_rows


def create_dataset(
    dataset, task_metadata, instruction_template, tokenizer, prompt_type, max_length
):
    """Create dataset.

    If prompt_type is None, it will create a dataloader based on the modalities of the task.
    if prompt_type is provided, it will create a dataloader for the specified prompt type.

    Args:
        dataset: The dataset to create a dataloader from.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt to create a dataloader for. If None, it will be inferred from the task metadata.
        tokenizer: The tokenizer to use.
        instruction_template: The instruction template function.
        max_length: Maximum sequence length in tokens.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        batch_size: The batch size for the dataloader.
        **kwargs: Additional arguments to pass to the dataloader creation functions.

    Returns:
        A tokenized dataset.
    """

    # input_type = task_metadata.get_modalities(prompt_type)

    # if input_type == ["text"]:  # text only

    if prompt_type == PromptType.document:
        filtered_ds = dataset.filter(_is_valid_corpus_row, desc=None)
    elif prompt_type == PromptType.query:
        if not isinstance(dataset["text"][0], list):
            filtered_ds = dataset.filter(_is_valid_query_row, desc=None)
        else:
            raise ValueError(f"Can't handle queries type queries for conversation")
    else:
        raise ValueError(f"Can't handle prompt type different from query or document")

    input_to_dict = partial(
        _build_prompt,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        prompt_type=prompt_type,
        task_metadata=task_metadata,
        eot_id=tokenizer.pad_token_id,
    )

    new_ds = filtered_ds.map(
        input_to_dict,
        batched=True,
        batch_size=10000,
        desc=None,
    )

    # Track removed IDs across all batches
    all_removed_ids = []
    def filter_wrapper(rows):
        keep_mask, removed_ids = _remove_long_sequences(rows, tokenizer, max_length)
        all_removed_ids.extend(removed_ids)
        return keep_mask

    new_ds = new_ds.filter(
        filter_wrapper,
        batched=True,
        batch_size=10000,
        desc=None,
    )
    # Store removed IDs as an attribute on the dataset
    new_ds.removed_ids = all_removed_ids

    return new_ds


def filter_paired_datasets_by_length(
    removed_query_ids,
    removed_positive_ids,
    queries_with_reps,
    positives_with_reps,
):
    """Filter query-positive pairs by length, maintaining 1-to-1 correspondence.
    Args:
        removed_query_ids: List of IDs of queries to remove
        removed_positive_ids: List of IDs of positives to remove
        queries_with_reps: Dictionary with query data including repetitions (must have "id" key)
        positives_with_reps: Dictionary with positive data including repetitions (must have "id" key)
        tokenizer: Tokenizer for measuring sequence length
        instruction_template: Function to build prompts
        task_metadata: Task metadata for prompt building
        max_length: Maximum sequence length in tokens
        rank: Process rank for logging (default: 0)

    Returns:
        Tuple of (filtered_unique_queries, filtered_unique_positives, filtered_queries, filtered_positives)
    """

    # Filter pairs: queries_with_reps[i] and positives_with_reps[i] are ALWAYS paired
    # Keep only if BOTH query AND positive are within max_length
    pair_valid_indices = []
    for i, (qid, pid) in enumerate(
        zip(queries_with_reps["id"], positives_with_reps["id"])
    ):

        if qid in removed_query_ids or pid in removed_positive_ids:
            continue
        else:
            pair_valid_indices.append(i)

    # Filter the paired datasets based on the valid indices
    filtered_queries = queries_with_reps.select(pair_valid_indices)
    filtered_positives = positives_with_reps.select(pair_valid_indices)

    return filtered_queries, filtered_positives


def instruction_template_embeddinggemma(prompt_type, task_metadata, row):

    text = row["text"]

    # we do not use  task specific instruction in embeddinggemma
    if prompt_type == PromptType.query:
        prompt = TASK_PROMPTS[task_metadata.type]

    elif prompt_type == PromptType.document:
        prompt = TASK_PROMPTS["Retrieval-document"]

        title = None
        if "title" in row and len(row["title"]) > 0:
            title = row["title"]

        if title is not None:
            prompt = TASK_PROMPTS["document"].format(title=title)

    return (prompt + text).strip()
