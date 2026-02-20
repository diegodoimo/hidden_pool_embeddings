"""
Shared loader functions for binary classification tasks.

Binary classification datasets have exactly two labels (0 and 1).  The loaders
here mirror the multi-way classification ones but add a convenience:
``label_texts`` (defined on each task) can optionally be prepended to the text
to enrich the training signal.

  * **Label-based / sampling** (``load_binary_classification_label_based``):
    Returns ``ClassificationRawData`` for on-the-fly contrastive pair creation.

  * **Hard-negative mining** (``load_binary_classification_hard_negatives``):
    Returns ``RetrievalRawData`` using the same conventions as retrieval
    loaders.  Positive pairs link texts that share the same binary label;
    ``corpus_dict`` entries carry a ``"label"`` key so the miner can
    restrict negative candidates to texts with the **opposite** label.
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
#  Internal helpers (reuse the classification pattern)
# ---------------------------------------------------------------------------


def _load_and_prepare(task, rank=None):
    """Load the HF dataset and return a pandas DataFrame with
    ``text``, ``label`` (int 0/1) columns.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``text``, ``label`` (int, 0 or 1).
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

    label_col = getattr(task, "label", "label")
    text_col = task.query_name

    df = dataset.select_columns([text_col, label_col]).to_pandas()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)

    # Ensure binary labels
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df.reset_index(drop=True, inplace=True)

    assert (
        df["label"].isin([0, 1]).all()
    ), f"Expected binary labels (0/1), got unique values: {df['label'].unique()}"

    dist.barrier()
    if rank == 0:
        print(f"Dataset loaded in {(time.time()-start)/60:.2f} min")
        n0 = (df["label"] == 0).sum()
        n1 = (df["label"] == 1).sum()
        print(
            f"  {return_formatted(len(df))} samples  "
            f"(label-0: {return_formatted(n0)}, label-1: {return_formatted(n1)})"
        )

    return df, rank


def _build_positive_pairs(df, max_num_queries=10**6, rank=0):
    """For every row, pick one same-label partner as a positive.

    Returns arrays of *indices* into ``df`` (query_idx, positive_idx).
    At most ``max_num_queries`` pairs are returned.
    """
    rng = np.random.default_rng(42)

    groups = {lbl: df.index[df["label"] == lbl].values for lbl in [0, 1]}

    query_indices = []
    positive_indices = []

    for lbl, members in groups.items():
        if len(members) < 2:
            continue
        for idx in members:
            partner = idx
            while partner == idx:
                partner = rng.choice(members)
            query_indices.append(idx)
            positive_indices.append(partner)

    query_indices = np.array(query_indices)
    positive_indices = np.array(positive_indices)

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


def load_binary_classification_label_based(
    task, rank=None, **kwargs
) -> ClassificationRawData:
    """Return texts + binary labels for on-the-fly contrastive sampling.

    The training loop is expected to create (anchor, positive, negative)
    tuples itself, using the labels to guarantee that negatives have the
    *opposite* label.
    """
    df, rank = _load_and_prepare(task, rank)

    ids = [f"text_{i}" for i in range(len(df))]

    if rank == 0:
        print(
            f"ClassificationRawData ready: {return_formatted(len(df))} texts, "
            f"2 labels (binary)"
        )

    return ClassificationRawData(
        texts=df["text"].tolist(),
        labels=df["label"].tolist(),
        ids=ids,
    )


def load_binary_classification_hard_negatives(
    task, max_num_queries=10**6, rank=None, **kwargs
) -> RetrievalRawData:
    """Produce ``RetrievalRawData`` for the hard-negative mining pipeline.

    * Every unique text becomes both a *query* and a *document*.
    * Positive pairs connect two texts that share the same binary label.
    * ``corpus_dict`` entries carry a ``"label"`` key (0 or 1) so the
      miner can restrict negative candidates to texts with the
      **opposite** label.
    """
    df, rank = _load_and_prepare(task, rank)

    if rank == 0:
        start = time.time()
        print("Building positive pairs & corpus...")

    # --- Deduplicate texts ---------------------------------------------------
    first_mask = ~df["text"].duplicated(keep="first")
    first_idx = first_mask[first_mask].index.values

    unique_ids = [f"text_{i}" for i in first_idx]
    unique_texts = df["text"].iloc[first_idx].reset_index(drop=True)
    unique_labels = df["label"].iloc[first_idx].reset_index(drop=True)

    n_unique = len(unique_ids)
    if rank == 0:
        print(
            f"  {return_formatted(n_unique)} unique texts "
            f"(from {return_formatted(len(df))} rows)"
        )

    # --- Build positive pairs ------------------------------------------------
    dedup_df = df.iloc[first_idx].reset_index(drop=True)
    query_idx, positive_idx = _build_positive_pairs(
        dedup_df,
        max_num_queries=max_num_queries,
        rank=rank,
    )

    query_ids = [unique_ids[i] for i in query_idx]
    positive_ids = [unique_ids[i] for i in positive_idx]

    if rank == 0:
        print(f"  {return_formatted(len(query_ids))} query→positive pairs")

    # --- Unique queries ------------------------------------------------------
    unique_query_mask = np.zeros(n_unique, dtype=bool)
    unique_query_mask[query_idx] = True
    unique_query_ids = [unique_ids[i] for i in np.nonzero(unique_query_mask)[0]]
    unique_query_texts = unique_texts.iloc[np.nonzero(unique_query_mask)[0]].tolist()

    # --- Corpus (all unique texts) -------------------------------------------
    document_ids = unique_ids
    document_texts = unique_texts.tolist()
    document_titles = None
    has_title = False

    referenced_positive_set = set(positive_ids)
    n_positives = len(referenced_positive_set)

    # --- corpus_dict with label info -----------------------------------------
    corpus_dict = {
        uid: {"text": txt, "label": int(lbl)}
        for uid, txt, lbl in zip(unique_ids, document_texts, unique_labels)
    }

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
