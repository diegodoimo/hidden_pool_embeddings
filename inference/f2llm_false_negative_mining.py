import os

# Disable Rust-level tokenizer parallelism to avoid deadlocks when the process
# is forked by DataLoader workers or DDP (the tokenizer is used inside collate_fn).
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.distributed as dist
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from tasks.load_datasets import load_task_data
from utils.helpers import print_memory_consumed

from inference.hard_negative_mining import _BaseMiner


class F2LLMValidator(_BaseMiner):
    """Annotate F2LLM parquet datasets with retrieval-quality signals.

    For each :class:`~tasks.f2llm_data_loaders.F2LLMParquetTask` this class:

    1. Tokenizes queries and corpus with the task's instruction template.
    2. Encodes queries and corpus, then retrieves the top-k most similar
       corpus documents for every query via :meth:`_run_encode_and_search`.
    3. Computes four annotation columns per ``(query_id, positive_id)`` pair
       via :meth:`_compute_f2llm_annotations` and appends them to the source
       parquet under model-prefixed column names.

    Column names use ``self.model_name`` with ``/`` replaced by ``__`` so
    that they form safe identifiers (e.g. ``Qwen__Qwen3-8B_false_negatives``).

    Columns written
    ---------------
    ``{model_name}_false_negatives``
        ``list<struct<doc_id: string, score: float32>>`` — corpus docs whose
        similarity exceeds 0.9 × the query-positive score; *score* is
        ``candidate_score / positive_score``.

    ``{model_name}_hard_negatives``
        ``list<string>`` — up to 24 corpus doc IDs that are neither the
        labeled positive nor a false negative (top of the similarity ranking).

    ``{model_name}_log_info_nce``
        ``float32`` — InfoNCE loss:
        ``logsumexp([s_pos, s_neg_1, …, s_neg_100]) − s_pos``.

    ``{model_name}_positive_rank``
        ``int32`` — 1-based rank of the labeled positive in the top-k list;
        set to ``top_k + 1`` when the positive is not found in the window.
    """

    def _compute_f2llm_annotations(
        self,
        top_scores,
        top_indices,
        corpus_ids,
        query_ids,
        positive_ids,
        unique_query_ids,
        query_positive_scores,
        unique_query_id_to_idx,
        n_hard_negatives=24,
        n_info_nce_negatives=100,
    ):
        """Compute all F2LLM annotation columns in a single pass over qrel pairs.

        Parameters
        ----------
        top_scores, top_indices       : np.ndarray [n_unique_queries, top_k]
        corpus_ids                    : list[str]
        query_ids, positive_ids       : list[str]  — one per qrel row
        unique_query_ids              : list[str]
        query_positive_scores         : list[float] — one per qrel row
        unique_query_id_to_idx        : dict[str, int]
        n_hard_negatives              : int  — max hard negatives to keep (default 24)
        n_info_nce_negatives          : int  — negatives used for InfoNCE (default 100)

        Returns
        -------
        false_negatives  : dict[(qid, pid)] → list[tuple[str, float]]
        hard_negatives   : dict[(qid, pid)] → list[str]
        log_info_nce     : dict[(qid, pid)] → float
        positive_rank    : dict[(qid, pid)] → int
        """
        array_ids = np.asarray(corpus_ids)
        total_queries = len(query_ids)
        top_k = top_scores.shape[1]

        false_negatives: dict = {}
        hard_negatives: dict = {}
        log_info_nce: dict = {}
        positive_rank: dict = {}
        n_fn_total = 0

        for qrel_idx, (q_id, p_id) in enumerate(zip(query_ids, positive_ids)):
            unique_q_idx = unique_query_id_to_idx[q_id]
            pos_score = query_positive_scores[qrel_idx]
            key = (q_id, p_id)
            if pos_score <= 0:
                # A zero or negative positive score makes the relative-relevance
                # ratio undefined and the thresholds degenerate.  Emit safe
                # defaults so every row still gets an entry in every column.
                false_negatives[key] = []       # no false negatives detectable
                hard_negatives[key] = []        # no hard negatives detectable
                log_info_nce[key] = float("nan")  # InfoNCE undefined
                positive_rank[key] = top_k + 1   # treat as not-found
                continue
            fn_threshold_false = 0.9*pos_score
            fn_threshold_hard = min(0.95*pos_score, 0.8) #as in f2llm

            all_scores = top_scores[unique_q_idx]   # shape [top_k], descending
            all_indices = top_indices[unique_q_idx]  # shape [top_k]
            candidate_ids = array_ids[all_indices]

            # ---- false negatives ------------------------------------------------
            fn_mask = (all_scores > fn_threshold_false) & (candidate_ids != p_id)
            fn_idx = np.where(fn_mask)[0]
            if fn_idx.size > 0:
                fn_ids = candidate_ids[fn_idx].tolist()
                # relative score: candidate_score / positive_score
                fn_rel = (all_scores[fn_idx] / pos_score).tolist()
                false_negatives[key] = list(zip(fn_ids, fn_rel))
                n_fn_total += fn_idx.size

            # ---- hard negatives -------------------------------------------------
            # Candidates that are not the positive and not false negatives,
            # taken from the top of the similarity-ranked list.
            hn_mask = (all_scores <= fn_threshold_hard) & (candidate_ids != p_id)
            hn_idx = np.where(hn_mask)[0][:n_hard_negatives]
            hard_negatives[key] = candidate_ids[hn_idx].tolist() if hn_idx.size > 0 else []

            # ---- log InfoNCE ----------------------------------------------------
            # Use positive + top n_info_nce_negatives non-positive docs.
            neg_mask = candidate_ids != p_id
            neg_scores = all_scores[neg_mask][:n_info_nce_negatives]
            logits = np.concatenate([[pos_score], neg_scores])
            # numerically stable logsumexp
            max_l = logits.max()
            log_denom = max_l + np.log(np.sum(np.exp(logits - max_l)))
            log_info_nce[key] = float(log_denom - pos_score)

            # ---- positive rank --------------------------------------------------
            pos_positions = np.where(candidate_ids == p_id)[0]
            positive_rank[key] = int(pos_positions[0]) + 1 if pos_positions.size > 0 else top_k + 1

        if self.rank == 0:
            n_with_fn = len(false_negatives)
            print(
                f"False negatives: {n_with_fn}/{total_queries} pairs have ≥1 FN "
                f"({n_fn_total} total FN document slots)"
            )

        return false_negatives, hard_negatives, log_info_nce, positive_rank

    def mine_false_negatives_f2llm(
        self,
        tasks,
        model,
        output_dir,
        batch_size=64,
        top_k=100,
    ):
        """Annotate F2LLM parquet files with retrieval-quality signals.

        For each F2LLMParquetTask in *tasks*:
          1. Load data via from_f2llm_parquet (reads task.parquet_path).
          2. Tokenize queries and corpus with the task's instruction template.
          3. Encode + search: retrieve the top-k most similar corpus documents
             for every query.
          4. Compute four annotation columns per (query_id, positive_id) pair
             via :meth:`_compute_f2llm_annotations`:
               - ``{model_name}_false_negatives`` — list of
                 ``{doc_id, score}`` structs for docs whose similarity exceeds
                 0.9 × the query-positive score; *score* is the ratio
                 candidate_score / positive_score.
               - ``{model_name}_hard_negatives`` — list of up to 24 corpus
                 doc IDs that are not false negatives and not the positive.
               - ``{model_name}_log_info_nce`` — InfoNCE loss computed from
                 the positive score and the top-100 non-positive doc scores:
                 ``logsumexp([s_pos, s_neg_1, …]) - s_pos``.
               - ``{model_name}_positive_rank`` — 1-based rank of the
                 labeled positive in the similarity-sorted top-k list.
          5. Read the source parquet, append the four columns, and write to
             *output_dir*.  The ``/`` in *model_name* is replaced with ``__``
             to form safe column-name prefixes.

        Only rank-0 performs I/O; all ranks participate in encoding/search.

        Parameters
        ----------
        tasks      : list[F2LLMParquetTask]
        model      : encoder (already on the correct device / DDP-wrapped)
        output_dir : str  — directory where annotated parquets are written
        batch_size : int
        top_k      : int  — number of top corpus documents retrieved per query
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for task in tasks:
            parquet_name = os.path.basename(task.parquet_path)

            if self.rank == 0:
                print(f"\n\nLOADING F2LLM DATASET: {parquet_name}\n")

            # ------------------------------------------------------------------
            # 1. Load and prepare dataset (tokenize queries + corpus).
            # ------------------------------------------------------------------
            (
                data_split,
                corpus_dict,
                query_dict,
                has_title,
                n_positives,
            ) = load_task_data(task)

            if self.rank == 0:
                print(f"\nPREPARING DATASET: {parquet_name}")

            dataset, corpus_dict = self.prepare_dataset(
                data_split=data_split,
                corpus_dict=corpus_dict,
                task_metadata=task.metadata,
                n_positives=n_positives,
            )
            del data_split

            dist.barrier()
            if self.rank == 0:
                print_memory_consumed(rank=self.rank)

            # ------------------------------------------------------------------
            # 2–3. Encode queries, search corpus, get raw top-k arrays.
            # ------------------------------------------------------------------
            (
                top_scores,
                top_indices,
                query_positive_scores,
                _unique_query_ids,
                _qrel_query_ids,
                _qrel_positive_ids,
                _corpus_ids,
                unique_query_id_to_idx,
                _,
            ) = self._run_encode_and_search(dataset, model, batch_size, top_k)

            # ------------------------------------------------------------------
            # 4. Compute all annotation columns (false negatives, hard
            #    negatives, log-InfoNCE loss, positive rank).
            # ------------------------------------------------------------------
            dist.barrier()
            if self.rank == 0:
                print("\nComputing F2LLM annotation columns")

            (
                false_negatives,
                hard_negatives,
                log_info_nce_dict,
                positive_rank_dict,
            ) = self._compute_f2llm_annotations(
                top_scores=top_scores,
                top_indices=top_indices,
                corpus_ids=_corpus_ids,
                query_ids=_qrel_query_ids,
                positive_ids=_qrel_positive_ids,
                unique_query_ids=_unique_query_ids,
                query_positive_scores=query_positive_scores,
                unique_query_id_to_idx=unique_query_id_to_idx,
            )

            del top_scores, top_indices, query_positive_scores, dataset
            torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # 5. Read source parquet, append four model-named columns, save.
            #    Only rank-0 does I/O; the barrier keeps all ranks in sync.
            # ------------------------------------------------------------------
            dist.barrier()
            if self.rank == 0:
                table = pq.read_table(task.parquet_path)

                q_ids = table.column("query_id").to_pylist()
                p_ids = table.column("positive_id").to_pylist()

                # Sanitise model name for use as a column-name prefix
                # (replace "/" so names are self-contained identifiers).
                col_prefix = self.model_name.replace("/", "__")

                # -- false_negatives: list of (doc_id, relative_score) structs --
                fn_rows = [
                    [
                        {"doc_id": d, "score": s}
                        for d, s in false_negatives.get((qid, pid), [])
                    ]
                    for qid, pid in zip(q_ids, p_ids)
                ]
                fn_type = pa.list_(
                    pa.struct([
                        pa.field("doc_id", pa.large_utf8()),
                        pa.field("score", pa.float32()),
                    ])
                )
                table = table.append_column(
                    f"{col_prefix}_false_negatives",
                    pa.array(fn_rows, type=fn_type),
                )

                # -- hard_negatives: list of doc_ids (top 24) --
                hn_rows = [
                    hard_negatives.get((qid, pid), [])
                    for qid, pid in zip(q_ids, p_ids)
                ]
                table = table.append_column(
                    f"{col_prefix}_hard_negatives",
                    pa.array(hn_rows, type=pa.list_(pa.large_utf8())),
                )

                # -- log_info_nce: scalar float per row --
                info_nce_rows = [
                    log_info_nce_dict.get((qid, pid), float("nan"))
                    for qid, pid in zip(q_ids, p_ids)
                ]
                table = table.append_column(
                    f"{col_prefix}_log_info_nce",
                    pa.array(info_nce_rows, type=pa.float32()),
                )

                # -- positive_rank: 1-based rank of the positive --
                rank_rows = [
                    positive_rank_dict.get((qid, pid), -1)
                    for qid, pid in zip(q_ids, p_ids)
                ]
                table = table.append_column(
                    f"{col_prefix}_positive_rank",
                    pa.array(rank_rows, type=pa.int32()),
                )

                out_path = os.path.join(output_dir, parquet_name)
                pq.write_table(table, out_path, compression="snappy")
                del table
                print(f"Saved annotated dataset → {out_path}")

            dist.barrier()
