from datasets import Dataset
import os

path = "/home/diego/Documents/area_science/ricerca/open/hidden_pool_embeddings/results/f2llm_data_no_instruct"

ds_name = "arguana"
path = os.path.join(path, f"{ds_name}.parquet")
ds = Dataset.from_parquet(path)


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
    # _print_ram("after loading HF dataset", rank)

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
    # _print_ram("after to_pandas (before del dataset)", rank)

    # Free the HF Arrow table — the data now lives in the pandas DataFrame.
    # For large datasets (e.g. BioASQ, 14 M rows) this reclaims ~10-20 GB.
    del dataset
    gc.collect()
    # _print_ram("after del dataset + gc", rank)
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
    # _print_ram("after deduplication", rank)

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
    # _print_ram("after del df + gc", rank)
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
    ), "filtered qrels contain positive IDs  in corpus"

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
    # _print_ram("after building CorpusDict", rank)
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
