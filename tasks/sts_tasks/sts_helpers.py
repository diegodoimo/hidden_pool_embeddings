"""
STS preprocessing functions.

These functions transform STS datasets (sentence1, sentence2, score) into the
retrieval-style format expected by ``from_one_hf_dataset``, so that a single
unified loader handles both STS and standard retrieval tasks.
"""

import pandas as pd
from datasets import Dataset

# Re-export from_one_hf_dataset so task files can import from here if desired.
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset  # noqa: F401


def make_sts_preprocessor(score_name="score", score_threshold=4.0):
    """Return a preprocessor that converts an STS dataset to retrieval format.

    The returned function has the signature expected by
    ``from_one_hf_dataset``'s ``task.preprocessor`` interface::

        preprocessor(dataset, query_name, positive_name) -> Dataset

    Processing steps (following Lee et al., 2025a):

    1. Filter to pairs whose similarity score is at least ``score_threshold``.
    2. Create bidirectional pairs: for every (s1, s2) where s1 != s2, also
       add (s2, s1).
    3. Emit a flat dataset with ``query_name`` and ``positive_name`` columns
       ready for ``from_one_hf_dataset``.

    Args:
        score_name: Column name containing the STS score.
        score_threshold: Minimum score to consider a pair as similar
            (default ``4.0``).
    """

    def sts_preprocessing(dataset, query_name, positive_name):
        # --- Convert to pandas -------------------------------------------------
        cols = [query_name, positive_name, score_name]
        df = dataset.select_columns(cols).to_pandas()

        # Filter by score threshold
        df = df[df[score_name] >= score_threshold].reset_index(drop=True)

        # Forward pairs
        pairs_fwd = df[[query_name, positive_name]].copy()

        # Reverse pairs (only where sentences differ)
        df_diff = df[df[query_name] != df[positive_name]]
        pairs_rev = df_diff[[positive_name, query_name]].copy()
        pairs_rev.columns = [query_name, positive_name]

        # Combine
        result = pd.concat([pairs_fwd, pairs_rev], ignore_index=True)

        return Dataset.from_pandas(result, preserve_index=False)

    return sts_preprocessing


# Pre-built default: covers STS12, STS22, STSBenchmark, and all other
# standard STS tasks where the score column is "score" and threshold == 4.
sts_preprocessor = make_sts_preprocessor()
