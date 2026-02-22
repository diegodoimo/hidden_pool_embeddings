"""
Shared loader functions for retrieval tasks.
These loaders are used by multiple retrieval tasks.
"""

import datasets as _datasets
from datasets import load_dataset
from typing import List, Optional
import gc
import os
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData, LazyCorpusDict, get_dict
from utils.helpers import return_formatted

_datasets.config.HF_DATASETS_TIMEOUT = 120


def _print_ram(label: str, rank: int = 0):
    """Print RAM usage on rank 0.

    Reports two complementary metrics:
      - VmRSS  (from /proc/self/status): RAM used by *this process* only
        (resident set size). Useful to see which step in the loader is
        responsible for memory growth.
      - available (from psutil): free RAM across the *entire system*,
        accounting for all processes and OS caches. Useful to know how
        close the machine is to OOM (especially with multiple DDP ranks).
    """
    if rank != 0:
        return
    # --- Per-process RSS (this process only) ---
    rss_gb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1024**2
                    break
    except FileNotFoundError:
        import resource

        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    # --- System-wide available RAM (all processes) ---
    avail_gb = None
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass

    parts = [f"[RAM] {label}:"]
    if rss_gb is not None:
        parts.append(f"process RSS {rss_gb:.2f} GB")
    if avail_gb is not None:
        parts.append(f"system available {avail_gb:.2f} GB")
    print("  ".join(parts))


def _load_hf_dataset(hf_name, config_name, split, revision=None):
    """Load a HuggingFace dataset, falling back to direct parquet loading when a
    revision is specified (needed for script-based datasets on datasets>=4.0).

    When *revision* is set (e.g. ``"refs/convert/parquet"``), the standard
    ``load_dataset(name=...)`` call cannot resolve configs because the parquet
    branch lacks the original loading-script metadata.  We instead construct
    ``hf://`` URLs pointing at the parquet files and load them directly.
    """
    if revision is None:
        if config_name:
            return load_dataset(hf_name, name=config_name, split=split)
        return load_dataset(hf_name, split=split)

    splits = [s.strip() for s in split.split("+")]
    base = f"hf://datasets/{hf_name}@{revision}"
    # Some HF parquet conversions use a directory layout
    # (e.g. config/train/0000.parquet) while others use a flat layout
    # (e.g. config/dataset-train.parquet).  Provide both patterns so that
    # at least one resolves for every subset.
    if config_name:
        data_files = {
            s: [
                f"{base}/{config_name}/{s}/*.parquet",
                f"{base}/{config_name}/*-{s}.parquet",
            ]
            for s in splits
        }
    else:
        data_files = {
            s: [
                f"{base}/{s}/*.parquet",
                f"{base}/*-{s}.parquet",
            ]
            for s in splits
        }
    return load_dataset("parquet", data_files=data_files, split=split)


def deduplicate(texts, prefix="query", titles=None):

    # Fast deduplication via pandas C-optimized hash tables.
    unique_mask = ~texts.duplicated(keep="first")
    unique_idx = unique_mask[unique_mask].index
    unique_texts = texts.iloc[unique_idx].reset_index(drop=True)
    unique_ids = [f"{prefix}_{i}" for i in unique_idx]

    if titles is not None:
        unique_titles = titles.iloc[unique_idx].reset_index(drop=True)
    else:
        unique_titles = None

    # Vectorized remapping: map each text to its first-occurrence index via pandas .map()
    # Build Series mapping text -> first occurrence index, then map over all rows at once
    text_to_first_idx = pd.Series(
        unique_idx.values, index=texts.iloc[unique_idx].values
    )
    # Generate remapped IDs: map all occurrences (including duplicates) to first occurrence ID
    all_ids = (f"{prefix}_" + texts.map(text_to_first_idx).astype(str)).tolist()

    return (
        all_ids,
        unique_ids,
        unique_idx,
        unique_texts,
        unique_titles,
    )


def from_one_hf_dataset(
    task, max_num_queries=None, rank=None, subtask=None
) -> RetrievalRawData:
    """
    Load data from a single HuggingFace dataset where queries and positives
    are in the same dataset with matching indices.

    Used by: NaturalQuestions, ALL_NLI, PAQ, ELI5, TriviaQA, COLIEE,
             S2ORC*, SPECTER, SentenceCompression, StackExchangeDup*, QQP, AmazonQA

    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank

    if rank == 0:
        start = time.time()
        print("Loading dataset...")

    subset_name = task.hf_subset
    if subtask is not None:
        assert task.hf_subset is None
        subset_name = subtask

    revision = getattr(task, "revision", None)
    dataset = _load_hf_dataset(task.hf_name, subset_name, task.split, revision=revision)
    #_print_ram("after loading HF dataset", rank)

    if task.preprocessor is not None:
        dataset = task.preprocessor(
            dataset,
            task.query_name,
            task.positive_name,
        )

    if task.decontaminator is not None:
        dataset = task.decontaminator(
            dataset,
            task.query_name,
            task.positive_name,
        )
    n_pairs = len(dataset)
    verbose = False
    if n_pairs > 10**6:
        verbose = True

    dist.barrier()
    if rank == 0 and verbose:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print(f"num elements in dataset: {return_formatted(n_pairs)}")
        print("building dataframes")

    # title_name is the canonical attribute for single-dataset tasks,
    # parallel to query_name / positive_name.
    title_col = getattr(task, "title_name", None)
    has_title = title_col is not None and title_col in dataset.column_names

    # Check for negatives to include in corpus
    has_negatives = task.negative_name is not None
    neg_col = None
    neg_title_col = None
    if has_negatives:
        neg_col = task.negative_name
        if neg_col not in dataset.column_names:
            has_negatives = False
            neg_col = None
        else:
            # Convention: if a column named <negative_name>_title exists,
            # use it for negative titles (created by preprocessors)
            candidate = neg_col + "_title"
            if candidate in dataset.column_names:
                neg_title_col = candidate

    # Convert Arrow -> pandas DataFrame in one shot (fast columnar conversion),
    # avoiding the slow path of dataset[col] (Python list) -> pd.Series.
    cols_to_load = [task.query_name, task.positive_name]
    if has_title:
        cols_to_load.append(title_col)
    if has_negatives:
        cols_to_load.append(neg_col)
        if neg_title_col is not None:
            cols_to_load.append(neg_title_col)
    df = dataset.select_columns(cols_to_load).to_pandas()
    #_print_ram("after to_pandas (before del dataset)", rank)

    # Free the HF Arrow table — the data now lives in the pandas DataFrame.
    # For large datasets (e.g. BioASQ, 14 M rows) this reclaims ~10-20 GB.
    del dataset
    gc.collect()
    #_print_ram("after del dataset + gc", rank)
    # --- OLD: dataset was not freed here, staying in memory alongside df ---

    # Keep as pandas Series — no .tolist() needed.
    # Dataset.from_dict() in dict_to_dataset() accepts Series directly,
    # so the round-trip Arrow → list → Arrow is avoided for 20M strings.

    # Drop rows where query or positive text is null (e.g. wikihow)
    null_mask = df[task.query_name].isna() | df[task.positive_name].isna()
    if null_mask.any():
        n_null = null_mask.sum()
        if rank == 0:
            print(f"Dropping {n_null} rows with null query or positive text")
        df = df[~null_mask].reset_index(drop=True)

    query_texts = df[task.query_name]
    positive_texts = df[task.positive_name]
    titles = None
    if has_title:
        titles = df[title_col]

    # Convert Arrow -> numpy arrays directly (fastest path)
    dist.barrier()
    if rank == 0 and verbose:
        print(f"preprocessing done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("finding unique queries and positives items...")

    query_ids, unique_query_ids, unique_query_idx, unique_query_texts, _ = deduplicate(
        query_texts, prefix="query"
    )
    (
        positive_ids,
        unique_positive_ids,
        unique_positive_idx,
        unique_positive_texts,
        unique_positive_titles,
    ) = deduplicate(positive_texts, prefix="doc", titles=titles)
    n_positives = len(unique_positive_ids)
    #_print_ram("after deduplication", rank)

    # Extract unique negative texts not already in positives
    neg_ids = []
    neg_texts = []
    neg_titles_list = None
    if has_negatives:
        if neg_title_col is not None:
            # Explode text and title columns in sync
            neg_df = df[[neg_col, neg_title_col]].copy()
            neg_df.columns = ["text", "title"]
            neg_df = neg_df.explode(["text", "title"]).dropna(subset=["text"])
            neg_df = neg_df.drop_duplicates(subset=["text"], keep="first").reset_index(
                drop=True
            )
            pos_texts_set = set(unique_positive_texts.tolist())
            neg_df = neg_df[~neg_df["text"].isin(pos_texts_set)].reset_index(drop=True)
            neg_ids = [f"neg_{i}" for i in range(len(neg_df))]
            neg_texts = neg_df["text"].tolist()
            neg_titles_list = neg_df["title"].tolist()
        else:
            neg_series = (
                df[neg_col]
                .explode()
                .dropna()
                .drop_duplicates(keep="first")
                .reset_index(drop=True)
            )
            pos_texts_set = set(unique_positive_texts.tolist())
            neg_series = neg_series[~neg_series.isin(pos_texts_set)].reset_index(
                drop=True
            )
            neg_ids = [f"neg_{i}" for i in range(len(neg_series))]
            neg_texts = neg_series.tolist()
        if rank == 0 and verbose:
            print(
                f"Found {return_formatted(len(neg_ids))} unique negatives not in positives"
            )

    # Free the DataFrame — column Series (query_texts, positive_texts, titles)
    # keep the underlying data alive via their own references.
    del df
    gc.collect()
    #_print_ram("after del df + gc", rank)
    # --- OLD: df was not freed here, staying in memory ---

    assert set(positive_ids).issubset(
        set(unique_positive_ids)
    ), "filtered qrels contain positive IDs not in corpus"

    # Apply query limiting only if needed
    if max_num_queries is not None and len(unique_query_ids) > max_num_queries:
        if rank == 0:
            start = time.time()
            print(
                f"Number of unique queries {return_formatted(len(unique_query_ids))} > {max_num_queries//10**6}M: limiting queries"
            )

        unique_query_texts = unique_query_texts[:max_num_queries]
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_idx = unique_query_idx[:max_num_queries]
        # Apply query limiting and reorganize documents
        (
            query_ids,
            positive_ids,
            unique_positive_ids,
            unique_positive_texts,
            unique_positive_titles,
            n_positives,
        ) = limit_number_of_queries(
            query_ids=query_ids,
            positive_ids=positive_ids,
            unique_query_idx=unique_query_idx,
            n_pairs=n_pairs,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            has_title=has_title,
        )

        if rank == 0 and verbose:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")

    assert set(positive_ids).issubset(
        set(unique_positive_ids)
    ), "filtered qrels contain positive IDs not in corpus"

    dist.barrier()
    if rank == 0 and verbose:
        print(f"remapping done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("generating corpus dict...")

    # Build document lists: positives first, then negatives (if any)
    if neg_ids:
        document_ids = list(unique_positive_ids) + neg_ids
        document_texts = list(unique_positive_texts) + neg_texts
        if has_title and unique_positive_titles is not None:
            if neg_titles_list is not None:
                document_titles = list(unique_positive_titles) + neg_titles_list
            else:
                document_titles = list(unique_positive_titles) + [""] * len(neg_ids)
        else:
            document_titles = unique_positive_titles
    else:
        document_ids = unique_positive_ids
        document_texts = unique_positive_texts
        document_titles = unique_positive_titles

    # Build corpus_dict with unique entries (bijective doc_id <-> document)
    # Use LazyCorpusDict to avoid materialising a dict-of-dicts, which
    # would duplicate all text data and add ~4-7 GB of Python object
    # overhead for multi-million-row datasets.

    # corpus_dict = LazyCorpusDict(
    #     ids=document_ids,
    #     texts=document_texts,
    #     titles=document_titles if has_title else None,
    # )

    # query_dict = LazyCorpusDict(
    #     ids=unique_query_ids,
    #     texts=unique_query_texts,
    # )
    
    # --- OLD: corpus_dict and query_dict were full Python dicts ---
    if has_title:
        corpus_dict = {
            id_: {"text": doc_text, "title": doc_title}
            for id_, doc_text, doc_title in zip(
                document_ids, document_texts, document_titles
            )
        }
    else:
        corpus_dict = {
            id_: {"text": doc_text}
            for id_, doc_text in zip(document_ids, document_texts)
        }
    
    query_dict = {
        id_: {"text": text} for id_, text in zip(unique_query_ids, unique_query_texts)
    }
    #_print_ram("after building CorpusDict", rank)
    assert set(document_ids) == set(corpus_dict.keys())
    dist.barrier()
    if rank == 0 and verbose:
        print(f"corpus dict built in {(time.time()-start)/60:.2f} min")

    if rank == 0:
        print(f"Found {return_formatted(len(unique_query_texts))} unique queries")
        print(
            f"Total number of query-positive pairs: {return_formatted(len(query_ids))}"
        )
        print(
            f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}"
        )
        print(
            f"Total unique documents in corpus: {return_formatted(len(document_ids))}"
        )

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        query_dict=query_dict,
        has_title=has_title,
        n_positives=n_positives,
    )


def from_multiple_hf_datasets(
    task, max_num_queries=10**6, rank=None, subtask=None
) -> RetrievalRawData:
    """
    Load data from multiple HuggingFace datasets (queries, corpus, qrels) using vectorized operations.
    This is the standard MTEB format with pandas-based vectorization for better performance.

    Uses the same unified format as from_one_hf_dataset:
    - Documents are unique with positives first
    - n_positives indicates how many positives are at the beginning

    When ``subtask`` is provided, the HF subset names are constructed by
    prefixing the subtask to the base names (e.g. subtask="arabic" with
    query_name="queries" → HF subset "arabic-queries").

    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
        subtask: Optional subtask prefix for constructing HF subset names
    """
    rank = dist.get_rank() if rank is None else rank
    verbose = False

    # Resolve HF subset names, optionally prefixed by subtask
    if subtask is not None:
        query_subset = f"{subtask}-{task.query_name}"
        corpus_subset = f"{subtask}-{task.positive_name}"
        qrels_subset = f"{subtask}-{task.qrels_name}"
    else:
        query_subset = task.query_name
        corpus_subset = task.positive_name
        qrels_subset = task.qrels_name

    if rank == 0:
        start = time.time()
        print("Loading datasets...")

    revision = getattr(task, "revision", None)
    qrels = _load_hf_dataset(task.hf_name, qrels_subset, task.split, revision=revision)
    # Use the base field name as the split (not the full subset name), because
    # hyphens in split names (e.g. "arabic-queries") are invalid in datasets.
    querys_ = _load_hf_dataset(
        task.hf_name, query_subset, task.query_name, revision=revision
    )
    corpus = _load_hf_dataset(
        task.hf_name, corpus_subset, task.positive_name, revision=revision
    )
    #_print_ram("after loading all HF datasets", rank)

    if len(qrels) > 10**6:
        verbose = True

    dist.barrier()
    if rank == 0 and verbose:
        print(f"Datasets loaded in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print(f"num elements in queries: {len(querys_)//10**3}k")
        print(f"num elements in qrels: {len(qrels)//10**3}k")
        print(f"num elements in corpus: {len(corpus)//10**3}k")
        print(f"Processing {len(querys_)} queries...")

    # Optional decontamination: filter qrels to remove train/eval overlap.
    # The decontaminator receives the raw qrels HF Dataset plus the actual
    # column names for the query-id and positive-id fields (parallel to how
    # from_one_hf_dataset calls decontaminator(dataset, query_field, positive_field)).
    if task.decontaminator is not None:
        if rank == 0:
            decon_start = time.time()
            print("Running decontaminator on qrels...")
        query_id_col = task.qrels_fields["query_id"]
        positive_id_col = task.qrels_fields["positive_id"]
        qrels = task.decontaminator(qrels, query_id_col, positive_id_col)
        dist.barrier()
        if rank == 0:
            print(f"Decontamination done in {(time.time()-decon_start)/60:.2f} min")
            print(f"num elements in qrels after decontamination: {len(qrels)//10**3}k")
            start = time.time()

    # Build queries dict
    queries_dict = get_dict(querys_, task.query_fields["id"], task.query_fields["text"])
    del querys_  # free HF Arrow table
    gc.collect()
    #_print_ram("after queries_dict + del querys_", rank)
    # --- OLD: querys_ was not freed here ---

    # Drop entries with null text (e.g. wikihow)
    null_query_ids = [k for k, v in queries_dict.items() if not v.get("text")]
    for k in null_query_ids:
        del queries_dict[k]

    dist.barrier()
    if rank == 0:
        if null_query_ids:
            print(f"Dropped {len(null_query_ids)} queries with null text")
    if rank == 0 and verbose:
        print(f"Queries processed in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print(f"Processing {len(corpus)} docs...")

    # Build corpus dict
    has_title = task.corpus_fields.get("title", None) is not None
    corpus_dict = get_dict(
        corpus,
        task.corpus_fields["id"],
        task.corpus_fields["text"],
        task.corpus_fields.get("title", None),
    )
    del corpus  # free HF Arrow table
    gc.collect()
    #_print_ram("after corpus_dict + del corpus", rank)
    # --- OLD: corpus HF dataset was not freed here ---

    # Drop entries with null text (e.g. wikihow)
    null_corpus_ids = [k for k, v in corpus_dict.items() if not v.get("text")]
    for k in null_corpus_ids:
        del corpus_dict[k]

    dist.barrier()
    if rank == 0:
        if null_corpus_ids:
            print(f"Dropped {len(null_corpus_ids)} corpus docs with null text")
    if rank == 0 and verbose:
        print(f"Corpus processed in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Processing qrels with vectorized operations...")

    # Convert qrels to pandas DataFrame for vectorized operations
    qrels_cols = [
        task.qrels_fields["query_id"],
        task.qrels_fields["positive_id"],
        task.qrels_fields["score"],
    ]
    df_qrels = qrels.select_columns(qrels_cols).to_pandas()

    # Rename columns for easier access
    df_qrels.columns = ["query_id", "positive_id", "score"]

    # Filter valid pairs: score >= 1, query_id in queries_dict, positive_id in corpus_dict
    valid_queries = df_qrels["query_id"].isin(queries_dict.keys())
    valid_positives = df_qrels["positive_id"].isin(corpus_dict.keys())
    valid_scores = df_qrels["score"] >= 1
    df_qrels = df_qrels[valid_queries & valid_positives & valid_scores].reset_index(
        drop=True
    )

    if rank == 0 and verbose:
        print(f"Found {len(df_qrels)} valid query-positive pairs")

    # Extract query and positive IDs for all pairs (these are string IDs from qrels)
    query_ids = df_qrels["query_id"].tolist()
    positive_ids = df_qrels["positive_id"].tolist()
    n_pairs = len(query_ids)

    dist.barrier()
    if rank == 0 and verbose:
        print(f"Found {n_pairs} valid query-positive pairs")
        print("Finding unique queries and positives...")

    # Get unique queries (preserving first occurrence order)
    unique_query_mask = ~df_qrels["query_id"].duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index.values
    unique_query_ids = df_qrels.loc[unique_query_mask, "query_id"].tolist()
    unique_query_texts = [queries_dict[qid]["text"] for qid in unique_query_ids]
    del queries_dict  # free the multiprocessed dict
    gc.collect()
    #_print_ram("after del queries_dict", rank)
    # --- OLD: queries_dict was not freed here ---

    # Get unique positives (preserving first occurrence order)
    unique_positive_mask = ~df_qrels["positive_id"].duplicated(keep="first")
    unique_positive_ids = df_qrels.loc[unique_positive_mask, "positive_id"].tolist()

    # Extract texts and titles for unique positives from corpus_dict
    if has_title:
        unique_positive_texts = [
            corpus_dict[pid]["text"] for pid in unique_positive_ids
        ]
        unique_positive_titles = [
            corpus_dict[pid]["title"] for pid in unique_positive_ids
        ]
    else:
        unique_positive_texts = [
            corpus_dict[pid]["text"] for pid in unique_positive_ids
        ]
        unique_positive_titles = None

    dist.barrier()
    if rank == 0 and verbose:
        print(
            f"Found {return_formatted(len(unique_query_ids))} unique queries before limiting"
        )
        print(
            f"Found {return_formatted(len(unique_positive_ids))} unique positives in qrels"
        )

    # Apply query limiting only if needed
    if max_num_queries is not None and len(unique_query_idx) > max_num_queries:
        if rank == 0:
            start = time.time()
            print(
                f"Number of unique queries {return_formatted(len(unique_query_idx))} > {max_num_queries//10**6}M: limiting queries"
            )

        # Limit unique queries to first max_num_queries
        unique_query_texts = unique_query_texts[:max_num_queries]
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_idx = unique_query_idx[:max_num_queries]

        # Use the unified limit_number_of_queries function
        (
            query_ids,
            positive_ids,
            document_ids,
            document_texts,
            document_titles,
            n_positives,
        ) = limit_number_of_queries(
            query_ids=query_ids,
            positive_ids=positive_ids,
            unique_query_idx=unique_query_idx,
            n_pairs=n_pairs,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            has_title=has_title,
        )

        # Add remaining corpus documents not in qrels
        seen_positive_ids_set = set(unique_positive_ids)
        all_document_ids_set = set(document_ids)
        remaining_corpus_ids = [
            doc_id
            for doc_id in corpus_dict.keys()
            if doc_id not in seen_positive_ids_set
            and doc_id not in all_document_ids_set
        ]

        if remaining_corpus_ids:
            document_ids.extend(remaining_corpus_ids)
            document_texts.extend(
                [corpus_dict[did]["text"] for did in remaining_corpus_ids]
            )
            if has_title:
                document_titles.extend(
                    [corpus_dict[did]["title"] for did in remaining_corpus_ids]
                )

        if rank == 0 and verbose:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
            print(
                f"Positives referenced by filtered pairs: {return_formatted(n_positives)}"
            )
            print(
                f"Total unique documents in corpus: {return_formatted(len(document_ids))}"
            )
    else:
        # No limiting needed, keep data as is
        # Build unified document lists: unique positives first, then remaining corpus docs
        seen_positive_ids_set = set(unique_positive_ids)
        remaining_doc_ids = [
            doc_id
            for doc_id in corpus_dict.keys()
            if doc_id not in seen_positive_ids_set
        ]

        # Concatenate: positives first, then remaining
        document_ids = unique_positive_ids + remaining_doc_ids
        document_texts = unique_positive_texts + [
            corpus_dict[did]["text"] for did in remaining_doc_ids
        ]

        if has_title:
            document_titles = unique_positive_titles + [
                corpus_dict[did]["title"] for did in remaining_doc_ids
            ]
        else:
            document_titles = None

        n_positives = len(unique_positive_ids)

        if rank == 0:
            print(f"Found {return_formatted(len(unique_query_ids))} unique queries")
            print(
                f"Total number of query-positive pairs: {return_formatted(len(query_ids))}"
            )
            print(
                f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}"
            )
            print(
                f"Total unique documents in corpus: {return_formatted(len(document_ids))}"
            )

    if rank == 0 and verbose:
        print(f"Qrels processing completed in {(time.time()-start)/60:.2f} min")

    # Assertions to ensure data consistency (same as from_one_hf_dataset)
    assert set(positive_ids).issubset(
        set(document_ids)
    ), "filtered qrels contain positive IDs not in document list"

    assert set(unique_positive_ids).issubset(
        set(document_ids)
    ), "unique positives not in document list"

    # Verify corpus_dict matches document_ids
    corpus_keys = set(corpus_dict.keys())
    doc_keys = set(document_ids)
    assert (
        corpus_keys == doc_keys
    ), f"corpus_dict keys mismatch with document_ids. Missing in corpus: {doc_keys - corpus_keys}, Missing in docs: {corpus_keys - doc_keys}"

    # Replace the heavy get_dict() Python dict-of-dicts with a lightweight
    # LazyCorpusDict that references the already-built document lists.
    del corpus_dict
    gc.collect()
    #_print_ram("after del corpus_dict (get_dict)", rank)
    corpus_dict = LazyCorpusDict(
        ids=document_ids,
        texts=document_texts,
        titles=document_titles if has_title else None,
    )

    query_dict = LazyCorpusDict(
        ids=unique_query_ids,
        texts=unique_query_texts,
    )
    #_print_ram("after building LazyCorpusDicts", rank)
    # --- OLD: corpus_dict was kept as a full dict-of-dicts from get_dict() ---
    # query_dict = {
    #     id_: {"text": text} for id_, text in zip(unique_query_ids, unique_query_texts)
    # }

    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=document_texts,
        document_ids=document_ids,
        document_titles=document_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        query_dict=query_dict,
        has_title=has_title,
        n_positives=n_positives,
    )


def limit_number_of_queries(
    query_ids,
    positive_ids,
    unique_query_idx,
    n_pairs,
    unique_positive_ids,
    unique_positive_texts,
    unique_positive_titles,
    has_title,
):
    """Optimized version using vectorized operations.

    Args:
        query_ids: List of query IDs for each pair
        positive_ids: List of positive IDs for each pair
        unique_query_idx: Array/list of indices corresponding to unique queries
        n_pairs: Total number of pairs
        unique_positive_ids: List of unique positive IDs
        unique_positive_texts: List of unique positive texts
        unique_positive_titles: List of unique positive titles (or None)
        has_title: Boolean indicating if titles exist
        max_queries: Maximum number of queries to keep (default: 1 million)

    Returns:
        Tuple of: (query_ids, positive_ids, document_ids, document_texts,
                   document_titles, n_positives)
    """

    # Convert inputs to numpy arrays once
    query_ids_arr = np.array(query_ids)
    positive_ids_arr = np.array(positive_ids)

    # Fast filtering: if unique_query_idx is already an array, use isin
    unique_query_idx_set = set(unique_query_idx)
    pair_indices = np.arange(n_pairs)
    valid_pair_mask = np.isin(pair_indices, list(unique_query_idx_set))

    query_ids_filtered = query_ids_arr[valid_pair_mask]
    positive_ids_filtered = positive_ids_arr[valid_pair_mask]

    # Get referenced positives
    referenced_positive_ids_set = set(positive_ids_filtered)

    # Convert to arrays for vectorized operations
    unique_positive_ids_arr = np.array(unique_positive_ids)
    unique_positive_texts_arr = np.array(unique_positive_texts)

    # Vectorized mask creation (still requires loop but on smaller set)
    referenced_mask = np.array(
        [pid in referenced_positive_ids_set for pid in unique_positive_ids_arr]
    )

    # Split using boolean indexing
    document_ids = np.concatenate(
        [
            unique_positive_ids_arr[referenced_mask],
            unique_positive_ids_arr[~referenced_mask],
        ]
    ).tolist()

    document_texts = np.concatenate(
        [
            unique_positive_texts_arr[referenced_mask],
            unique_positive_texts_arr[~referenced_mask],
        ]
    ).tolist()

    if has_title:
        unique_positive_titles_arr = np.array(unique_positive_titles)
        document_titles = np.concatenate(
            [
                unique_positive_titles_arr[referenced_mask],
                unique_positive_titles_arr[~referenced_mask],
            ]
        ).tolist()
    else:
        document_titles = None

    n_positives = referenced_mask.sum()

    return (
        query_ids_filtered.tolist(),
        positive_ids_filtered.tolist(),
        document_ids,
        document_texts,
        document_titles,
        n_positives,
    )
