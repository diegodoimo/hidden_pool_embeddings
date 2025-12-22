from mteb.types import PromptType
from functools import partial

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


def instruction_template_qwen3(prompt_type, task_metadata, row) -> str:
    text = row["text"]

    if prompt_type == PromptType.query:
        if task_metadata.prompt is not None:
            instruction = task_metadata.prompt["query"]
            prompt = f"Instruct: {instruction.strip()}\nQuery: {text.strip()}"
        else:
            prompt = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text.strip()}"

    elif prompt_type == PromptType.document:

        title = row.get("title") or ""  # Use .get and default to empty string

        if len(title) > 0:
            prompt = f"{title} {text.strip()}"
        else:
            prompt = text.strip()  # Just return the text if no title

    return prompt


def instruction_template_embeddinggemma(prompt_type, task_metadata, row):
    text = row["text"]

    # we do not use task specific instruction in embeddinggemma
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


def _build_prompt(
    tokenizer,
    instruction_template,
    prompt_type,
    task_metadata,
    rows: dict[str, str],
) -> dict[str, str]:

    text_prompts = [instruction_template(prompt_type, task_metadata, row=row) for row in rows]
    tokens = [tokenizer.encode(prompt) for prompt in text_prompts]

    new_rows = {
        "id": rows["id"],
        "input_ids": tokens,
        "text": text_prompts,
        "body": rows["text"],
    }

    return new_rows


def create_dataset(
    dataset,
    task_metadata,
    instruction_template,
    tokenizer,
    prompt_type,
):
    """Create dataset.

    If prompt_type is None, it will create a dataloader based on the modalities of the task.
    if prompt_type is provided, it will create a dataloader for the specified prompt type.

    Args:
        dataset: The dataset to create a dataloader from.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt to create a dataloader for. If None, it will be inferred from the task metadata.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        batch_size: The batch size for the dataloader.
        **kwargs: Additional arguments to pass to the dataloader creation functions.

    Returns:
        A tokenized dataset.
    """

    input_type = task_metadata.get_modalities(prompt_type)

    if input_type == ["text"]:  # text only

        if prompt_type == PromptType.document:
            filtered_ds = dataset.filter(_is_valid_corpus_row)
        elif prompt_type == PromptType.query:
            if not isinstance(dataset["text"][0], list):
                filtered_ds = dataset.filter(_is_valid_query_row)
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
        )

        new_ds = filtered_ds.map(
            input_to_dict,
            batched=True,
            batch_size=1000,
        )

        return new_ds

    raise ValueError(f"Can't handle queries type {input_type}")
