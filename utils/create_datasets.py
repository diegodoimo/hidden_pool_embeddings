from mteb.types import PromptType
from functools import partial
from datasets import Dataset, concatenate_datasets
import torch.distributed as dist
import time
import numpy as np
from datasets import disable_progress_bars

import os
import glob
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from functools import partial
import mteb

# taken from embeddinggemma
# https://github.com/huggingface/transformers/blob/bdee0889714e9cb3e53d3b1b2a626919479d356c/src/transformers/models/gemma3/convert_gemma3_weights.py#L700C1-L715C10
# TASK_PROMPTS = {
#     "query": "task: search result | query: ",
#     "document": "title: {title} | text: {text}",
#     "BitextMining": "task: search result | query: ",
#     "Clustering": "task: clustering | query: ",
#     "Classification": "task: classification | query: ",
#     "InstructionRetrieval": "task: code retrieval | query: ",
#     "MultilabelClassification": "task: classification | query: ",
#     "PairClassification": "task: sentence similarity | query: ",
#     "Reranking": "task: search result | query: ",
#     "Retrieval": "task: search result | query: ",
#     "Retrieval-query": "task: search result | query: ",
#     "Retrieval-document": "title: none | text: ",
#     "STS": "task: sentence similarity | query: ",
#     "Summarization": "task: summarization | query: ",
# }

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

# Example subset of 10 datasets from results/datasets_negatives/qwen3_600m leaf folders.
# Use as datasets_subset=QWEN3_600M_10DATASET_SUBSET to restrict training to these.
QWEN3_600M_DATASET_SUBSET = [
    # "retrieval/general_retrieval/msmarco",
    "retrieval/general_retrieval/nfcorpus",
    "retrieval/general_retrieval/arguana",
    "retrieval/domain_specific_qa/fiqa2018",
    "retrieval/open_domain_qa/naturalquestions",
    "retrieval/open_domain_qa/squad",
    "retrieval/fact_verification/scifact",
    "retrieval/summarization/xsum",
    "sts/stsbenchmark",
    "nli/snli",
]


# MTEB 20-task subset (mteb_20task_subset_selection.md) - minimizes eval time while preserving category averages
TASK_DICT = {
    "mteb_eng_v2_reduced": [
        "SCIDOCS",
        "CQADupstackGamingRetrieval",
        "CQADupstackUnixRetrieval",
        "HotpotQAHardNegatives",
        # "TRECCOVID",
        # "TwentyNewsgroupsClustering.v2",
        # "BiorxivClusteringP2P.v2",
        # "MedrxivClusteringS2S.v2",
        # "StackExchangeClustering.v2",
        # "AskUbuntuDupQuestions",
        # "BIOSSES",
        "STS17",
        "STS12",
        # "AmazonCounterfactualClassification",
        # "MassiveScenarioClassification",
        # "TweetSentimentExtractionClassification",
        # "MTOPDomainClassification",
        # "TwitterSemEval2015",
        # "SprintDuplicateQuestions",
        # "SummEvalSummarization.v2",
    ],
}


def get_eval_tasks(eval_set):
    """Return list of MTEB task objects for evaluation."""

    if eval_set == "mteb_multilingual_v2":
        benchmark = mteb.get_benchmark("MTEB(Multilingual, v2)")
        tasks = list(benchmark.tasks)
    elif eval_set == "mteb_eng_v2":
        benchmark = mteb.get_benchmark("MTEB(eng, v2)")
        tasks = list(benchmark.tasks)
    elif eval_set == "mteb_eng_v2_20":
        task_names = TASK_DICT["mteb_eng_v2_20"]
        tasks = [mteb.get_task(name) for name in task_names]
    else:
        raise ValueError(f"Unknown eval_set: {eval_set}")
    return tasks


@dataclass
class TrainTaskMetadata:
    type: str
    prompt: str = None


FOLDER_TO_TASK = {
    "retrieval/general_retrieval": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a web search query, retrieve relevant passages that answer the query",
    ),
    "retrieval/domain_specific_qa": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a question, retrieve passages that answer the question",
    ),
    "retrieval/open_domain_qa": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a question, retrieve passages that answer the question",
    ),
    "retrieval/fact_verification": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a claim, retrieve documents that support or refute the claim",
    ),
    "retrieval/paraphrase_detection": TrainTaskMetadata(
        type="STS",
        prompt="Retrieve semantically similar text",
    ),
    "retrieval/scientific_doc_retrieval": TrainTaskMetadata(
        type="Retrieval",
        prompt="Given a scientific paper title, retrieve the paper abstract",
    ),
    "retrieval/summarization": TrainTaskMetadata(
        type="Summarization",
        prompt="Given a summary, retrieve the original document",
    ),
    "nli": TrainTaskMetadata(
        type="Classification",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
    ),
    "sts": TrainTaskMetadata(
        type="STS",
        prompt="Retrieve semantically similar text",
    ),
}

DEFAULT_TASK = TrainTaskMetadata(
    type="Retrieval",
    prompt="Given a web search query, retrieve relevant passages that answer the query",
)


def _infer_task_metadata(parquet_path, base_dir):
    """Infer task metadata from the directory structure."""
    rel = os.path.relpath(os.path.dirname(parquet_path), base_dir)
    parts = rel.replace(os.sep, "/").split("/")
    for depth in range(len(parts), 0, -1):
        key = "/".join(parts[:depth])
        if key in FOLDER_TO_TASK:
            return FOLDER_TO_TASK[key]
    return DEFAULT_TASK


def _load_parquet_safe(path):
    """Load a parquet file as an HF Dataset with fallback for metadata issues."""
    try:
        return Dataset.from_parquet(path)
    except (TypeError, Exception):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        table = table.replace_schema_metadata({})
        return Dataset(table)


def _str_to_int_id(s: str) -> int:
    """Deterministic hash of a string to a positive 63-bit integer."""
    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)


def instruction_template_qwen3(prompt_type, task_metadata, text, title="") -> str:
    # text = row["text"]

    if prompt_type == PromptType.query:
        if task_metadata.prompt is not None:
            if isinstance(task_metadata.prompt, dict):
                instruction = task_metadata.prompt["query"]
            else:
                instruction = task_metadata.prompt
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


def instruction_template_embeddinggemma(prompt_type, task_metadata, text, title=""):

    # text = row["text"]

    # we do not use  task specific instruction in embeddinggemma
    if prompt_type == PromptType.query:
        instruction = TASK_PROMPTS[task_metadata.type]
        prompt = f"{instruction.strip()} {text.strip()}"

    elif prompt_type == PromptType.document:

        if len(title) > 0:
            instruction = TASK_PROMPTS["document"].format(title=title)
        else:
            instruction = TASK_PROMPTS["Retrieval-document"]

        prompt = f"{instruction.strip()} {text.strip()}"

    return prompt


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
    # --- SLOW original implementation (kept for reference) ---
    # Two sources of inefficiency:
    # 1. `removed_query_ids` and `removed_positive_ids` are plain Python lists, so
    #    `qid not in removed_query_ids` is an O(n) linear scan on every iteration,
    #    making the overall loop O(n * m) where n=14M pairs and m=number of removed ids.
    # 2. The pure-Python for-loop over 14M rows has large per-iteration overhead
    #    (bytecode dispatch, dynamic type checks, list.append) compared to vectorized C code.
    #
    # pair_valid_indices = []
    # for i, (qid, pid) in enumerate(
    #     zip(qrels_dataset["query_id"], qrels_dataset["positive_id"])
    # ):
    #     if qid not in removed_query_ids and pid not in removed_positive_ids:
    #         pair_valid_indices.append(i)
    # filtered_qrels = qrels_dataset.select(pair_valid_indices)
    # return filtered_qrels
    # ----------------------------------------------------------
    if not removed_query_ids and not removed_positive_ids:
        return qrels_dataset

    # --- Previous approach (kept for reference) ---
    # pd.Series(qrels_dataset["query_id"]) first decodes the internal Arrow column
    # into 14M Python string objects, then wraps them in a Series — that
    # materialisation alone took ~3.75 min for 14M rows.
    # The fix: operate directly on the underlying Arrow table so the data
    # never leaves C++ memory.
    import pandas as pd

    removed_query_set = set(removed_query_ids)
    removed_positive_set = set(removed_positive_ids)
    query_valid = (
        ~pd.Series(qrels_dataset["query_id"]).isin(removed_query_set).to_numpy()
    )
    positive_valid = (
        ~pd.Series(qrels_dataset["positive_id"]).isin(removed_positive_set).to_numpy()
    )
    keep_mask = query_valid & positive_valid
    valid_indices = np.where(keep_mask)[0].tolist()
    return qrels_dataset.select(valid_indices)
    # -----------------------------------------------

    # Access the underlying Arrow table — no Python object creation for 14M rows.
    # pc.is_in runs entirely in C++ against an Arrow hash table, then
    # table.filter() applies the boolean mask without going through Python.

    # SOME ERROR AFFTECT THE BELOW CODE

    # import pyarrow as pa
    # import pyarrow.compute as pc
    # from datasets import Dataset

    # arrow_table = qrels_dataset.data.table

    # query_keep = pc.invert(
    #     pc.is_in(
    #         arrow_table.column("query_id"),
    #         value_set=pa.array(list(removed_query_ids)),
    #     )
    # )
    # positive_keep = pc.invert(
    #     pc.is_in(
    #         arrow_table.column("positive_id"),
    #         value_set=pa.array(list(removed_positive_ids)),
    #     )
    # )
    # keep_mask = pc.and_(query_keep, positive_keep)

    # return Dataset(arrow_table.filter(keep_mask))


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


def _build_prompts_hard_negatives_batch(
    examples,
    tokenizer,
    instruction_template,
    task_metadata,
    num_hard_negatives,
):
    """Build prompts for (query, positive, negatives) using create_datasets._build_prompt.

    Mirrors the create_dataset flow: uses _build_prompt for query, positive, and negatives.
    """
    batch_size = len(examples["query_text"])
    eot_id = tokenizer.pad_token_id

    # Query prompts (mirror create_dataset: map with _build_prompt)
    q_rows = {
        "text": examples["query_text"],
        "id": examples.get("query_id", [str(i) for i in range(batch_size)]),
    }
    query_result = _build_prompt(
        q_rows,
        tokenizer,
        instruction_template,
        PromptType.query,
        task_metadata,
        eot_id,
    )
    query_prompts = query_result["prompt"]
    total_length = [len(q) for q in query_prompts]

    # Positive prompts
    pos_titles = examples.get("positive_title", None)
    if pos_titles is None:
        pos_titles = [""] * batch_size
    p_rows = {
        "text": examples["positive_text"],
        "id": examples["positive_id"],
        "title": pos_titles,
    }
    pos_result = _build_prompt(
        p_rows,
        tokenizer,
        instruction_template,
        PromptType.document,
        task_metadata,
        eot_id,
    )
    positive_prompts = pos_result["prompt"]

    total_length = [len(p) + l for p, l in zip(positive_prompts, total_length)]

    # Negative prompts: flatten, build, unflatten
    all_neg_texts = []
    all_neg_ids = []
    all_neg_titles = []
    for i in range(batch_size):
        neg_texts = examples["negative_text"][i][:num_hard_negatives]
        neg_titles_col = examples.get("negative_title", None)
        if neg_titles_col and neg_titles_col[i]:
            neg_titles = neg_titles_col[i][:num_hard_negatives]
        else:
            neg_titles = [""] * len(neg_texts)
        all_neg_texts.extend(neg_texts)
        all_neg_ids.extend([f"{i}_{j}" for j in range(len(neg_texts))])
        all_neg_titles.extend(neg_titles)

    n_rows = {"text": all_neg_texts, "id": all_neg_ids, "title": all_neg_titles}
    neg_result = _build_prompt(
        n_rows,
        tokenizer,
        instruction_template,
        PromptType.document,
        task_metadata,
        eot_id,
    )
    neg_prompts_flat = neg_result["prompt"]
    # Unflatten
    idx = 0
    negative_prompts = []
    neg_length = []
    for i in range(batch_size):
        n = min(len(examples["negative_text"][i]), num_hard_negatives)
        negative_prompts.append(neg_prompts_flat[idx : idx + n])
        neg_length.append(sum(len(neg) for neg in neg_prompts_flat[idx : idx + n]))
        idx += n

    total_length = [p + n for p, n in zip(neg_length, total_length)]

    return {
        "query_prompt": query_prompts,
        "positive_prompt": positive_prompts,
        "negative_prompts": negative_prompts,
        "total_length": total_length,
    }


def create_dataset(
    dataset,
    task_metadata,
    instruction_template,
    tokenizer,
    prompt_type,
    max_length,
    verbose=False
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

    if rank == 0 and verbose:
        print(f"prompt constructed in {(time.time()-start)/60:.2f}min")
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

    new_ds = new_ds.filter(filter_wrapper, batched=True, batch_size=10000)

    if rank == 0 and verbose:
        print(f"dataset filtered in {(time.time()-start)/60:.2f}min")
    # Store removed IDs as an attribute on the dataset
    new_ds.removed_long = all_removed_long_ids
    new_ds.removed_empty = all_removed_empty_ids
    new_ds.removed_ids = all_removed_long_ids + all_removed_empty_ids
    assert len(new_ds.removed_ids) == len(all_removed_long_ids) + len(
        all_removed_empty_ids
    )

    return new_ds


# ---------------------------------------------------------------------------
# Hard negatives dataset loading and tokenization
# ---------------------------------------------------------------------------


def create_hard_negatives_datasets(
    base_dir,
    num_hard_negatives,
    tokenizer,
    instruction_template,
    rank=0,
    datasets_subset: Optional[List[str]] = None,
):
    """Load and tokenize all hard-negative parquet datasets under *base_dir*.

    Mirrors the logic of create_datasets.create_dataset:
    1. Map: build prompts using _build_prompt (via _build_prompts_hard_negatives_batch)
    2. Filter: remove long sequences using _remove_long_sequences_hard_negatives
    3. Map: tokenize the prompts

    Args:
        base_dir: Root directory containing dataset subdirs with data.parquet
        num_hard_negatives: Number of hard negatives per example
        tokenizer: HuggingFace tokenizer
        instruction_template: Callable for building instruction prompts
        max_query_len: Max query token length
        max_passage_len: Max passage token length
        rank: Process rank (0 = main, for logging)
        datasets_subset: Optional list of dataset names (relative paths from base_dir)
            to restrict loading. Names should match leaf folders, e.g.
            "retrieval/general_retrieval/msmarco", "sts/stsbenchmark".
            Use QWEN3_600M_10DATASET_SUBSET for a 10-dataset example.

    Returns a single concatenated HF Dataset sorted by total sequence length
    (longest first) for length-balanced batching.
    """

    parquet_files = sorted(
        glob.glob(os.path.join(base_dir, "**", "data.parquet"), recursive=True)
    )

    if datasets_subset is not None:
        subset_set = set(datasets_subset)
        parquet_files = [
            p
            for p in parquet_files
            if os.path.relpath(os.path.dirname(p), base_dir) in subset_set
        ]
        if rank == 0:
            print(
                f"Restricted to {len(parquet_files)} datasets (subset of {len(datasets_subset)} requested)"
            )

    if rank == 0:
        print(f"Found {len(parquet_files)} datasets under {base_dir}")

    all_datasets = []
    for i, path in enumerate(parquet_files):
        task_metadata = _infer_task_metadata(path, base_dir)
        ds_name = os.path.relpath(os.path.dirname(path), base_dir)

        if rank == 0:
            print(f"  Loading {ds_name} {i}/{len(parquet_files)}...")

        ds = _load_parquet_safe(path)

        # Step 1: Build prompts (mirror create_dataset: map with _build_prompt)
        start = time.time()
        build_fn = partial(
            _build_prompts_hard_negatives_batch,
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            task_metadata=task_metadata,
            num_hard_negatives=num_hard_negatives,
        )
        ds = ds.map(build_fn, batched=True, batch_size=10000,)

        # Step 3: Add lengths and dataset_name for sorting (tokenization done in collate)
        # len_fn = partial(
        #     _add_lengths_and_dataset_name,
        #     tokenizer=tokenizer,
        #     max_query_len=max_query_len,
        #     max_passage_len=max_passage_len,
        #     num_hard_negatives=num_hard_negatives,
        #     dataset_name=ds_name,
        # )
        # ds = ds.map(len_fn, batched=True, batch_size=1000)

        # Ensure dataset_name column exists (parquet files may or may not have it)
        if "dataset_name" not in ds.column_names:
            ds = ds.add_column("dataset_name", [ds_name] * len(ds))

        # Keep only columns needed for collate
        ds = ds.sort("total_length", reverse=True)
        cols_to_keep = {
            "query_prompt",
            "positive_prompt",
            "negative_prompts",
            "positive_id",
            "query_id",
            "dataset_name",
            "total_length"
        }
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        ds = ds.remove_columns(cols_to_remove)
        all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)

    if rank == 0:
        total_tokens = np.sum(combined["total_length"])
        print(f"Total training examples: {len(combined)/10**6:.2f}M")
        print(f"Total tokens: {total_tokens/10**9:.2f}B")


    return combined
