"""
NLI preprocessing functions.

These functions transform NLI datasets (premise, hypothesis, label) into the
retrieval-style format expected by ``from_one_hf_dataset``, so that a single
unified loader handles both NLI and standard retrieval tasks.
"""

import pandas as pd
from datasets import Dataset

# Re-export from_one_hf_dataset so task files can import from here if desired.
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset  # noqa: F401


def make_nli_preprocessor(label_name="label", entailment_label=0):
    """Return a preprocessor that converts an NLI dataset to retrieval format.

    The returned function has the signature expected by
    ``from_one_hf_dataset``'s ``task.preprocessor`` interface::

        preprocessor(dataset, query_name, positive_name) -> Dataset

    Processing steps:

    1. Filter out invalid labels (e.g. ``-1`` in SNLI).
    2. Retain only premises that have at least one entailed hypothesis.
    3. For each such premise, sample one entailed hypothesis as the positive.
    4. Collect all remaining hypotheses (both entailed and non-entailed) for
       that premise into a list-valued ``"negative"`` column so they end up in
       the corpus and can be mined as hard negatives.

    Args:
        label_name: Column name containing the NLI label.
        entailment_label: Integer value of the entailment class (default ``0``).
    """

    def nli_preprocessing(dataset, query_name, positive_name):
        # --- Convert to pandas -------------------------------------------------
        cols = [query_name, positive_name, label_name]
        df = dataset.select_columns(cols).to_pandas()

        # Filter out invalid labels (e.g. -1 in SNLI)
        df = df[df[label_name] >= 0].reset_index(drop=True)

        # Keep only premises with at least one entailment
        entailment_mask = df[label_name] == entailment_label
        valid_premises = set(df.loc[entailment_mask, query_name].unique())
        df = df[df[query_name].isin(valid_premises)].reset_index(drop=True)

        # Sample one entailed hypothesis per premise as the positive
        df_entail = df[df[label_name] == entailment_label]
        sampled = (
            df_entail.groupby(query_name)[positive_name]
            .apply(lambda x: x.sample(n=1, random_state=42).iloc[0])
            .reset_index()
        )
        sampled.columns = [query_name, positive_name]

        # Collect *all* hypotheses per premise (will become negatives)
        all_hyp = df.groupby(query_name)[positive_name].apply(list).reset_index()
        all_hyp.columns = [query_name, "_all_hypotheses"]

        # Merge and build the negative list (everything except the positive)
        result = sampled.merge(all_hyp, on=query_name)
        result["negative"] = result.apply(
            lambda row: [h for h in row["_all_hypotheses"] if h != row[positive_name]],
            axis=1,
        )
        result = result.drop(columns=["_all_hypotheses"])

        return Dataset.from_pandas(result, preserve_index=False)

    return nli_preprocessing


# Pre-built default: covers SNLI, MNLI, ANLI, XNLI, and all other standard
# NLI tasks where the label column is "label" and entailment == 0.
nli_preprocessor = make_nli_preprocessor()
