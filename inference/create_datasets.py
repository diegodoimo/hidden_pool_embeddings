from mteb.types import PromptType
from functools import partial
from datasets import Dataset
import torch.distributed as dist
import time
import numpy as np
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


def _is_valid_row(text: str) -> bool:
    """Check if a dataset row has non-empty text content."""
    if not text or not text.strip():
        return False
    return True


# def _remove_long_sequences(rows, tokenizer, max_length):
#     """Remove rows where the tokenized prompt exceeds max_length.

#     Returns:
#         tuple: (keep_mask, removed_long_ids, removed_empty_ids)
#             - keep_mask: list of booleans indicating which rows to keep
#             - removed_long_ids: list of IDs that were removed due to length
#             - removed_empty_ids: list of IDs that were removed due to being empty
#     """
#     keep_mask = []
#     removed_long_ids = []
#     removed_empty_ids = []

#     # rows is a batched dictionary: {"id": [...], "text": [...], "prompt": [...]}
#     for i, (prompt, text) in enumerate(zip(rows["prompt"], rows["text"])):

#         if not _is_valid_row(text):
#             keep_mask.append(False)
#             removed_empty_ids.append(rows["id"][i])
#             continue

#         # Fast path: if char length is very short, definitely keep
#         if len(prompt) <= max_length:
#             keep_mask.append(True)
#         else:
#             # Must tokenize to check actual token length
#             token_length = len(tokenizer.encode(prompt, add_special_tokens=False))
#             if token_length > max_length:
#                 keep_mask.append(False)
#                 removed_long_ids.append(rows["id"][i])
#             else:
#                 keep_mask.append(True)

#     return keep_mask, removed_long_ids, removed_empty_ids


def _remove_long_sequences(rows, tokenizer, max_length):
    """Remove rows where the tokenized prompt exceeds max_length."""

    texts = np.array(rows["text"])
    ids = np.array(rows["id"])
    prompts = rows["prompt"]

    # Vectorized empty check
    valid_text_mask = np.array([bool(text and text.strip()) for text in texts])
    removed_empty_ids = ids[~valid_text_mask].tolist()

    # Pre-filter by character length (fast heuristic)
    char_lengths = np.array([len(p) for p in prompts])
    definitely_valid = char_lengths <= max_length
    needs_tokenization = (char_lengths > max_length) & valid_text_mask

    # Batch tokenize only the ones that need it
    prompts_to_check = [prompts[i] for i in np.where(needs_tokenization)[0]]

    if prompts_to_check:
        # Batch tokenization - MUCH faster than loop
        tokenized = tokenizer(
            prompts_to_check,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        token_lengths = np.array([len(ids) for ids in tokenized["input_ids"]])
        too_long_mask = token_lengths > max_length

        # Map back to original indices
        check_indices = np.where(needs_tokenization)[0]
        removed_long_indices = check_indices[too_long_mask]
        removed_long_ids = ids[removed_long_indices].tolist()
    else:
        removed_long_ids = []
        removed_long_indices = np.array([], dtype=int)

    # Final keep mask: start from the character-length heuristic, then
    # re-add items that needed tokenization but turned out to be within
    # the token limit (their char length exceeded max_length but token
    # count did not).
    keep_mask = valid_text_mask & definitely_valid
    if prompts_to_check:
        check_indices = np.where(needs_tokenization)[0]
        within_limit = check_indices[~too_long_mask]
        keep_mask[within_limit] = True

    return keep_mask.tolist(), removed_long_ids, removed_empty_ids


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
    rank = dist.get_rank()
    if "text" not in dataset.column_names:
        raise ValueError("Column 'text' not found in dataset")

    if isinstance(dataset["text"][0], list):
        raise ValueError("Can't handle queries type queries for conversation")

    start = time.time()

    input_to_dict = partial(
        _build_prompt,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        prompt_type=prompt_type,
        task_metadata=task_metadata,
        eot_id=tokenizer.pad_token_id,
    )
    new_ds = dataset.map(input_to_dict, batched=True, batch_size=10000)

    if rank == 0:
        print(f"prompt constructed in {(time.time()-start)/60}min")
        start = time.time()

    all_removed_long_ids = []
    all_removed_empty_ids = []

    def filter_wrapper(rows):
        keep_mask, removed_long, removed_empty = _remove_long_sequences(
            rows, tokenizer, max_length
        )
        all_removed_long_ids.extend(removed_long)
        all_removed_empty_ids.extend(removed_empty)
        return keep_mask

    new_ds = new_ds.filter(
        filter_wrapper,
        batched=True,
    )

    if rank == 0:
        print(f"dataset filtered in {(time.time()-start)/60}min")
    # Store removed IDs as an attribute on the dataset
    new_ds.removed_long = all_removed_long_ids
    new_ds.removed_empty = all_removed_empty_ids
    new_ds.removed_ids = all_removed_long_ids + all_removed_empty_ids
    assert len(new_ds.removed_ids) == len(all_removed_long_ids) + len(
        all_removed_empty_ids
    )

    return new_ds


def filter_qrels_by_length(
    removed_query_ids,
    removed_positive_ids,
    qrels_dataset,
):
    """Filter qrels dataset by removing pairs where either query or positive was removed due to length.

    Args:
        removed_query_ids: List of IDs of queries to remove
        removed_positive_ids: List of IDs of positives to remove
        qrels_dataset: Dataset with query_id and positive_id columns

    Returns:
        Filtered qrels dataset
    """

    # Filter pairs: Keep only if BOTH query AND positive are within max_length
    pair_valid_indices = []
    for i, (qid, pid) in enumerate(
        zip(qrels_dataset["query_id"], qrels_dataset["positive_id"])
    ):
        if qid not in removed_query_ids and pid not in removed_positive_ids:
            pair_valid_indices.append(i)

    # Filter the qrels dataset based on the valid indices
    filtered_qrels = qrels_dataset.select(pair_valid_indices)

    return filtered_qrels


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
