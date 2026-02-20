"""
Shared loader functions for multi-way classification tasks.

These loaders convert classification datasets (text + label) into formats
suitable for contrastive training:

  * **Sampling** (`load_multiway_classification_sampling`):
    Returns ``ClassificationRawData`` – a lightweight container that the training
    pipeline can consume directly to build pairs/negatives on-the-fly.

  * **Hard-negative mining** (`load_multiway_classification_hard_negatives`):
    Returns ``RetrievalRawData`` – the same format used by retrieval loaders so
    the existing ``HardNegativesMiner`` can be reused.  Every text becomes both
    a *query* and a *document*; positive pairs link texts that share a label,
    and the ``corpus_dict`` entries carry a ``"label"`` key so the miner can
    restrict negative candidates to texts with a **different** label.
"""

import datasets as _datasets
from datasets import load_dataset
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData, ClassificationRawData
from utils.helpers import return_formatted

_datasets.config.HF_DATASETS_TIMEOUT = 120


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _get_label_col(task):
    """Return the dataset column name that holds the label.

    Some tasks use ``task.label`` (column name in HF dataset), others
    use ``task.label_name``.  Fall back to ``"label"`` if neither is set.
    """
    col = getattr(task, "label_name", None) or getattr(task, "label", None) or "label"
    return col


def _load_and_prepare(task, rank=None):
    """Load the HF dataset and return a pandas DataFrame with
    ``text``, ``label`` and (optionally) ``title`` columns plus a
    ``label_encoder`` mapping original label values → integer ids.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``text``, ``label`` (original), ``label_id`` (int).
        Optionally ``title`` if the task defines ``title_name``.
    label_encoder : dict
        Mapping ``original_label_value → int``.
    rank : int
    """
    rank = dist.get_rank() if rank is None else rank

    if rank == 0:
        start = time.time()
        print("Loading dataset...")

    trust_remote_code = getattr(task, "trust_remote_code", False)
    hf_subset = getattr(task, "hf_subset", None)

    if hf_subset:
        dataset = load_dataset(
            task.hf_name,
            name=hf_subset,
            split=task.split,
            trust_remote_code=trust_remote_code,
        )
    else:
        dataset = load_dataset(
            task.hf_name,
            split=task.split,
            trust_remote_code=trust_remote_code,
        )

    label_col = _get_label_col(task)
    text_col = task.query_name
    title_col = getattr(task, "title_name", None)

    cols = [text_col, label_col]
    if title_col and title_col in dataset.column_names:
        cols.append(title_col)
    else:
        title_col = None

    df = dataset.select_columns(cols).to_pandas()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    if title_col:
        df.rename(columns={title_col: "title"}, inplace=True)

    # Drop rows with missing text / label
    df.dropna(subset=["text", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode labels to consecutive integers
    unique_labels = sorted(df["label"].unique(), key=str)
    label_encoder = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df["label_id"] = df["label"].map(label_encoder).astype(int)

    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        print(
            f"  {return_formatted(len(df))} samples, "
            f"{len(unique_labels)} unique labels"
        )

    return df, label_encoder, rank


def _build_positive_pairs(df, max_num_queries=10**6, rank=0):
    """For every row, pick one same-label partner as a positive.

    Returns arrays of *indices* into ``df`` (query_idx, positive_idx).
    At most ``max_num_queries`` pairs are returned.
    """
    rng = np.random.default_rng(42)

    # Group row indices by label_id
    groups = df.groupby("label_id").apply(
        lambda g: g.index.values, include_groups=False
    )

    query_indices = []
    positive_indices = []

    for label_id, members in groups.items():
        if len(members) < 2:
            continue  # Skip labels with a single example
        for idx in members:
            # Pick a *different* member with the same label
            partner = idx
            while partner == idx:
                partner = rng.choice(members)
            query_indices.append(idx)
            positive_indices.append(partner)

    query_indices = np.array(query_indices)
    positive_indices = np.array(positive_indices)

    # Limit to max_num_queries
    if max_num_queries is not None and len(query_indices) > max_num_queries:
        if rank == 0:
            print(
                f"Limiting pairs from {return_formatted(len(query_indices))} "
                f"to {return_formatted(max_num_queries)}"
            )
        sel = rng.choice(len(query_indices), size=max_num_queries, replace=False)
        sel.sort()
        query_indices = query_indices[sel]
        positive_indices = positive_indices[sel]

    return query_indices, positive_indices


# ---------------------------------------------------------------------------
#  Public loaders
# ---------------------------------------------------------------------------


def load_multiway_classification_sampling(
    task, rank=None, **kwargs
) -> ClassificationRawData:
    """Return texts + integer labels for on-the-fly contrastive sampling.

    The training loop is expected to create (anchor, positive, negative)
    tuples itself, using the labels to ensure negatives come from a
    different class.
    """
    df, label_encoder, rank = _load_and_prepare(task, rank)

    has_title = "title" in df.columns
    texts = df["text"].tolist()
    if has_title:
        titles = df["title"].tolist()
        texts = [
            f"{title}. {text}" if title else text for title, text in zip(titles, texts)
        ]

    ids = [f"text_{i}" for i in range(len(df))]

    if rank == 0:
        print(
            f"ClassificationRawData ready: {return_formatted(len(texts))} texts, "
            f"{len(label_encoder)} labels"
        )

    return ClassificationRawData(
        texts=texts,
        labels=df["label_id"].tolist(),
        ids=ids,
    )


def load_multiway_classification_hard_negatives(
    task, max_num_queries=10**6, rank=None, **kwargs
) -> RetrievalRawData:
    """Produce ``RetrievalRawData`` suitable for the hard-negative mining
    pipeline.

    * Every unique text becomes both a *query* and a *document*.
    * Positive pairs connect two texts that share the same label.
    * ``corpus_dict`` entries carry a ``"label"`` key so the miner can
      restrict negative candidates to texts whose label differs from
      the query's label.

    The resulting structure is intentionally identical to what retrieval
    loaders return, so ``HardNegativesMiner`` works unchanged.
    """
    df, _, rank = _load_and_prepare(task, rank)

    if rank == 0:
        start = time.time()
        print("Building positive pairs & corpus...")

    has_title = "title" in df.columns

    # --- Deduplicate texts ---------------------------------------------------
    if has_title:
        dedup_key = df["text"] + " ||| " + df["title"]
    else:
        dedup_key = df["text"]

    first_mask = ~dedup_key.duplicated(keep="first")
    first_idx = first_mask[first_mask].index.values  # indices of first-occurrence rows

    # Build id arrays: same text maps to same id (first-occurrence id)
    unique_ids = [f"text_{i}" for i in first_idx]
    unique_texts = df["text"].iloc[first_idx].reset_index(drop=True)
    unique_labels = df["label_id"].iloc[first_idx].reset_index(drop=True)
    unique_titles = (
        df["title"].iloc[first_idx].reset_index(drop=True) if has_title else None
    )

    n_unique = len(unique_ids)

    if rank == 0:
        print(
            f"  {return_formatted(n_unique)} unique texts "
            f"(from {return_formatted(len(df))} rows)"
        )

    # --- Build positive pairs ------------------------------------------------
    # Work on the deduplicated set: for each unique text find a same-label partner
    query_idx, positive_idx = _build_positive_pairs(
        df.iloc[first_idx].reset_index(drop=True),
        max_num_queries=max_num_queries,
        rank=rank,
    )

    query_ids = [unique_ids[i] for i in query_idx]
    positive_ids = [unique_ids[i] for i in positive_idx]

    if rank == 0:
        print(f"  {return_formatted(len(query_ids))} query→positive pairs")

    # --- Unique queries (subset of unique texts that appear as queries) -------
    unique_query_mask = np.zeros(n_unique, dtype=bool)
    unique_query_mask[query_idx] = True
    unique_query_ids = [unique_ids[i] for i in np.nonzero(unique_query_mask)[0]]
    unique_query_texts = unique_texts.iloc[np.nonzero(unique_query_mask)[0]].tolist()

    # --- Build corpus (= all unique texts) ------------------------------------
    document_ids = unique_ids
    document_texts = unique_texts.tolist()
    document_titles = unique_titles.tolist() if has_title else None

    # n_positives: texts referenced as positives are at unknown positions
    # in the flat list, but since every text is both a query and a candidate
    # the whole corpus is the search space.  We set n_positives to the
    # number of unique texts that actually appear as a positive in the qrels.
    referenced_positive_set = set(positive_ids)
    n_positives = len(referenced_positive_set)

    # --- corpus_dict with label info ------------------------------------------
    if has_title:
        corpus_dict = {
            uid: {"text": txt, "title": ttl, "label": int(lbl)}
            for uid, txt, ttl, lbl in zip(
                unique_ids, document_texts, document_titles, unique_labels
            )
        }
    else:
        corpus_dict = {
            uid: {"text": txt, "label": int(lbl)}
            for uid, txt, lbl in zip(unique_ids, document_texts, unique_labels)
        }

    # query_dict (also with label for convenience)
    query_dict = {
        uid: {"text": corpus_dict[uid]["text"], "label": corpus_dict[uid]["label"]}
        for uid in unique_query_ids
    }

    dist.barrier()
    if rank == 0:
        print(f"Corpus & pairs built in {(time.time()-start)/60:.2f} min")
        print(f"  {return_formatted(len(unique_query_ids))} unique queries")
        print(f"  {return_formatted(len(document_ids))} documents in corpus")
        print(f"  {return_formatted(n_positives)} documents referenced as positives")

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
