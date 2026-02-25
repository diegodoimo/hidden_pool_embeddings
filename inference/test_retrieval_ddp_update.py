import mteb
import torch
from torch.utils.data import DataLoader
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType

from functools import partial
import torch.distributed as dist
from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores
import torch.nn.functional as F
from mteb._evaluators.retrieval_metrics import make_score_dict
from utils.sorted_sampler import LenghtSortedSampler
from inference.helpers import (
    search,
    encode,
    collate_fn_with_padding,
    abs_task_preprocessing,
    last_token_pool,
    mean_pool,
)
from inference.hard_negative_mining import estimate_chunk_sizes
from inference.create_datasets import create_dataset
from utils.helpers import _print_ram
from collections import defaultdict
import time
import itertools
import numpy as np
from datasets import Dataset as HFDataset

from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import clone
from mteb.abstasks.multilabel_classification import _evaluate_classifier
from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
from mteb._evaluators.pair_classification_evaluator import PairClassificationDistances
from scipy.stats import pearsonr, spearmanr
from mteb._evaluators.text.summarization_evaluator import SummarizationEvaluator



class evaluate_retrieval:

    def __init__(
        self,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
        new_inference_mode=True,
        pool_fn=last_token_pool,
        add_special_tokens=False,
        append_eos=True,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tokenizer = tokenizer
        # self.task_names = [task.metadata.name for task in tasks]
        self.tasks = tasks
        self.padding_side = padding_side
        self.new_inference_mode = new_inference_mode
        self.pool_fn = pool_fn
        self.add_special_tokens = add_special_tokens
        self.append_eos = append_eos

        t1 = time.time()
        _print_ram(label="before loading datasets", rank=self.rank)
        self.datasets = self.prepare_datasets(instruction_template)
        dist.barrier()
        _print_ram(label="after loading datastes", rank=self.rank)
        if self.rank ==0:
            print(f"datsets prepared in {(time.time()-t1)/60:.2f} min")

    def _get_max_split_size_from_hub(self, task):
        """Query HF Hub for the largest split size (rows) without downloading data.

        Inspects all dataset configs (e.g. corpus/queries for retrieval tasks)
        and returns (max_rows, description_string).  Only metadata is fetched.
        """
        from datasets import get_dataset_config_names, get_dataset_config_info

        ds_kwargs = dict(task.metadata.dataset)
        path = ds_kwargs.get("path")
        if not path:
            return 0, ""
        revision = ds_kwargs.get("revision")

        try:
            configs = get_dataset_config_names(path, revision=revision)
        except Exception as e:
            if self.rank == 0:
                print(f"  WARNING: could not list configs for {path}: {e}")
            return 0, ""

        max_rows = 0
        max_info = ""
        for config in configs:
            try:
                info = get_dataset_config_info(
                    path, config_name=config, revision=revision
                )
                if info.splits:
                    for split_name, split_info in info.splits.items():
                        if split_info.num_examples > max_rows:
                            max_rows = split_info.num_examples
                            max_info = f"{config}/{split_name}"
            except Exception:
                continue

        return max_rows, max_info

    def prepare_datasets(
        self, instruction_template, max_passage_len=4096, max_samples=10_000_000
    ):

        datasets = {}
        n_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            task_name = task.metadata.name
            task_type = task.metadata.type

            if self.rank == 0:
                print(f"processing {task_name} {i}/{n_tasks}")

            if self.rank == 0:
                print(f"preparing {task_name} (type: {task_type})")

            eval_splits = task.metadata.eval_splits
            if "test" in eval_splits:
                eval_split = "test"
            else:
                eval_split = eval_splits[-1]
            if self.rank == 0 and len(eval_splits) > 1:
                print(f"  multiple eval_splits {eval_splits}, using '{eval_split}'")

            max_rows, size_info = self._get_max_split_size_from_hub(task)
            if max_rows > max_samples:
                if self.rank == 0:
                    print(
                        f"  SKIPPING {task_name}: largest split has {max_rows} rows "
                        f"({size_info}, limit={max_samples})"
                    )
                continue
            if self.rank == 0 and max_rows > 0:
                print(f"  largest split: {max_rows} rows ({size_info})")

            task.load_data()

            if task_type == "Retrieval":
                self._prepare_retrieval(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "PairClassification":
                self._prepare_pair_classification(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "MultilabelClassification":
                self._prepare_multilabel_classification(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Clustering":
                self._prepare_clustering(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Classification":
                self._prepare_classification(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "STS":
                self._prepare_sts(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Summarization":
                self._prepare_summarization(
                    task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Reranking":
                self._prepare_retrieval(
                    task, task_name, eval_split, instruction_template, datasets
                )
            else:
                if self.rank == 0:
                    print(f"  WARNING: unsupported task type '{task_type}', skipping")
            _print_ram(label="dataset loaded", rank=self.rank)

        return datasets

    # -------------------------------------------------------------------------
    # Dataset preparation per task type
    # -------------------------------------------------------------------------

    def _prepare_retrieval(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        task.convert_v1_dataset_format_to_v2()
        data_split, hf_subset = abs_task_preprocessing(task, eval_split)

        data_split["relevant_docs"], data_split["queries"] = (
            _filter_queries_without_positives(
                data_split["relevant_docs"], data_split["queries"]
            )
        )

        queries_dataset = create_dataset(
            dataset=data_split["queries"],
            task_metadata=task.metadata,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
            max_length=4096,
        )

        corpus_dataset = create_dataset(
            dataset=data_split["corpus"],
            task_metadata=task.metadata,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.document,
            max_length=4096,
        )

        relevant_docs = data_split["relevant_docs"]
        removed_query_ids = (
            set(queries_dataset.removed_ids) if queries_dataset.removed_ids else set()
        )
        removed_corpus_ids = (
            set(corpus_dataset.removed_ids) if corpus_dataset.removed_ids else set()
        )
        if removed_query_ids or removed_corpus_ids:
            if self.rank == 0:
                print(
                    f"  filtered {len(removed_query_ids)} queries, "
                    f"{len(removed_corpus_ids)} corpus docs (>4096 tokens or empty)"
                )
            relevant_docs = {
                qid: {
                    did: s
                    for did, s in docs.items()
                    if did not in removed_corpus_ids
                }
                for qid, docs in relevant_docs.items()
                if qid not in removed_query_ids
            }
            relevant_docs = {
                qid: docs for qid, docs in relevant_docs.items() if docs
            }

        datasets[task_name] = {
            "dataset": {
                "queries": queries_dataset,
                "corpus": corpus_dataset,
                "relevant_docs": relevant_docs,
            },
            "task_specific_scores": task.task_specific_scores,
            "ignore_identical_ids": task.ignore_identical_ids,
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
        }

    def _prepare_text_dataset(self, texts, task_metadata, instruction_template, max_length=8192):
        """Create a prompt-augmented HF dataset from raw texts for encoding.

        Returns:
            (ds, removed_indices): the filtered dataset and the set of original
            integer positions that were removed (too long or empty).
        """
        dataset = HFDataset.from_dict(
            {"id": [str(i) for i in range(len(texts))], "text": texts}
        )
        ds = create_dataset(
            dataset=dataset,
            task_metadata=task_metadata,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            prompt_type=PromptType.query,
            max_length=max_length,
        )
        removed_indices = set(int(x) for x in ds.removed_ids) if ds.removed_ids else set()
        if removed_indices and self.rank == 0:
            print(
                f"  WARNING: {len(removed_indices)}/{len(texts)} texts filtered "
                f"(>{max_length} tokens or empty)"
            )
        return ds, removed_indices

    @staticmethod
    def _build_index_remap(n_original, removed_set):
        """Build old-index -> new-index mapping after removing items."""
        old_to_new = {}
        new_idx = 0
        for old_idx in range(n_original):
            if old_idx not in removed_set:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        return old_to_new

    def _prepare_pair_classification(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        data_split, hf_subset = abs_task_preprocessing(task, eval_split)

        # v1 compatibility: some datasets store all pairs in a single row
        if task.metadata.modalities == ["text"]:
            if isinstance(data_split, HFDataset) and len(data_split) == 1:
                data_split = data_split[0]

        input1_col = task.input1_column_name
        input2_col = task.input2_column_name
        label_col = task.label_column_name

        sentence1 = list(data_split[input1_col])
        sentence2 = list(data_split[input2_col])
        labels = list(data_split[label_col])

        # Deduplicate texts for efficient encoding
        all_sentences = sentence1 + sentence2
        unique_texts, text_to_idx = [], {}
        for text in all_sentences:
            h = hash(text)
            if h not in text_to_idx:
                text_to_idx[h] = len(unique_texts)
                unique_texts.append(text)

        indices1 = [text_to_idx[hash(s)] for s in sentence1]
        indices2 = [text_to_idx[hash(s)] for s in sentence2]

        if self.rank == 0:
            n_dedup = len(all_sentences) - len(unique_texts)
            print(
                f"  {n_dedup}/{len(all_sentences)} duplicate texts deduplicated"
            )

        texts_ds, removed = self._prepare_text_dataset(
            unique_texts, task.metadata, instruction_template
        )

        if removed:
            old_to_new = self._build_index_remap(len(unique_texts), removed)
            valid_mask = [
                indices1[i] not in removed and indices2[i] not in removed
                for i in range(len(indices1))
            ]
            indices1 = [old_to_new[indices1[i]] for i in range(len(indices1)) if valid_mask[i]]
            indices2 = [old_to_new[indices2[i]] for i in range(len(indices2)) if valid_mask[i]]
            labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]
            if self.rank == 0:
                n_removed = sum(1 for v in valid_mask if not v)
                print(f"  {n_removed} pairs removed due to filtered texts")

        datasets[task_name] = {
            "dataset": {
                "texts": texts_ds,
                "indices1": indices1,
                "indices2": indices2,
                "labels": labels,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    def _prepare_multilabel_classification(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        from typing import cast
        from copy import copy
        from mteb.types import HFSubset
        from datasets import DatasetDict

        task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)
        hf_subsets = (
            copy(task.hf_subsets) if task.hf_subsets else list(task.dataset.keys())
        )
        assert len(hf_subsets) == 1, hf_subsets
        hf_subset = hf_subsets[0]

        if hf_subset not in task.dataset and hf_subset == "default":
            ds = task.dataset
        else:
            ds = task.dataset[hf_subset]

        input_col = task.input_column_name
        label_col = task.label_column_name

        if isinstance(ds, DatasetDict):
            ds = ds.select_columns([input_col, label_col])

        train_data = ds[task.train_split]
        test_data = ds[eval_split]

        # Subsample test set to 2000 as MTEB does
        try:
            if len(test_data) > 2000:
                split_ds = test_data.train_test_split(
                    test_size=2000, seed=42, stratify_by_column=label_col
                )
                test_data = split_ds["test"]
        except ValueError:
            if self.rank == 0:
                print("  Could not stratify test subsample, using full test set")

        train_texts = list(train_data[input_col])
        test_texts = list(test_data[input_col])
        train_labels = list(train_data[label_col])
        test_labels = list(test_data[label_col])

        if self.rank == 0:
            print(f"  train: {len(train_texts)}, test: {len(test_texts)} samples")

        train_ds, train_removed = self._prepare_text_dataset(
            train_texts, task.metadata, instruction_template
        )
        test_ds, test_removed = self._prepare_text_dataset(
            test_texts, task.metadata, instruction_template
        )

        if train_removed:
            train_labels = [l for i, l in enumerate(train_labels) if i not in train_removed]
        if test_removed:
            test_labels = [l for i, l in enumerate(test_labels) if i not in test_removed]

        datasets[task_name] = {
            "dataset": {
                "train_texts": train_ds,
                "test_texts": test_ds,
                "train_labels": train_labels,
                "test_labels": test_labels,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    def _prepare_clustering(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        data_split, hf_subset = abs_task_preprocessing(task, eval_split)

        input_col = task.input_column_name
        label_col = task.label_column_name

        sentences = list(data_split[input_col])
        labels = list(data_split[label_col])

        if (
            task.max_document_to_embed is not None
            and task.max_fraction_of_documents_to_embed is not None
        ):
            raise ValueError(
                "Both max_document_to_embed and max_fraction_of_documents_to_embed are set"
            )

        if task.max_fraction_of_documents_to_embed is not None:
            max_docs = min(
                len(sentences),
                int(task.max_fraction_of_documents_to_embed * len(sentences)),
            )
            indices = task.rng_state.sample(range(len(sentences)), k=max_docs)
            sentences = [sentences[i] for i in indices]
            labels = [labels[i] for i in indices]
        elif task.max_document_to_embed is not None:
            max_docs = min(len(sentences), task.max_document_to_embed)
            indices = task.rng_state.sample(range(len(sentences)), k=max_docs)
            sentences = [sentences[i] for i in indices]
            labels = [labels[i] for i in indices]

        if self.rank == 0:
            print(f"  {len(sentences)} samples for clustering")

        texts_ds, removed = self._prepare_text_dataset(
            sentences, task.metadata, instruction_template
        )

        if removed:
            labels = [l for i, l in enumerate(labels) if i not in removed]

        datasets[task_name] = {
            "dataset": {
                "texts": texts_ds,
                "labels": labels,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    def _prepare_classification(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        from typing import cast
        from copy import copy
        from mteb.types import HFSubset
        from datasets import DatasetDict

        task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)
        hf_subsets = (
            copy(task.hf_subsets) if task.hf_subsets else list(task.dataset.keys())
        )
        assert len(hf_subsets) == 1, hf_subsets
        hf_subset = hf_subsets[0]

        if hf_subset not in task.dataset and hf_subset == "default":
            ds = task.dataset
        else:
            ds = task.dataset[hf_subset]

        input_col = task.input_column_name
        label_col = task.label_column_name

        if isinstance(ds, DatasetDict):
            ds = ds.select_columns([input_col, label_col])

        train_data = ds[task.train_split]
        test_data = ds[eval_split]

        train_texts = list(train_data[input_col])
        test_texts = list(test_data[input_col])
        train_labels = list(train_data[label_col])
        test_labels = list(test_data[label_col])

        if self.rank == 0:
            print(f"  train: {len(train_texts)}, test: {len(test_texts)} samples")

        train_ds, train_removed = self._prepare_text_dataset(
            train_texts, task.metadata, instruction_template
        )
        test_ds, test_removed = self._prepare_text_dataset(
            test_texts, task.metadata, instruction_template
        )

        if train_removed:
            train_labels = [l for i, l in enumerate(train_labels) if i not in train_removed]
        if test_removed:
            test_labels = [l for i, l in enumerate(test_labels) if i not in test_removed]

        datasets[task_name] = {
            "dataset": {
                "train_texts": train_ds,
                "test_texts": test_ds,
                "train_labels": train_labels,
                "test_labels": test_labels,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    def _prepare_sts(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        data_split, hf_subset = abs_task_preprocessing(task, eval_split)

        col1, col2 = task.column_names

        sentence1 = list(data_split[col1])
        sentence2 = list(data_split[col2])
        raw_scores = list(data_split["score"])
        normalized_scores = [task._normalize(s) for s in raw_scores]

        all_sentences = sentence1 + sentence2
        unique_texts, text_to_idx = [], {}
        for text in all_sentences:
            h = hash(text)
            if h not in text_to_idx:
                text_to_idx[h] = len(unique_texts)
                unique_texts.append(text)

        indices1 = [text_to_idx[hash(s)] for s in sentence1]
        indices2 = [text_to_idx[hash(s)] for s in sentence2]

        if self.rank == 0:
            n_dedup = len(all_sentences) - len(unique_texts)
            print(
                f"  {n_dedup}/{len(all_sentences)} duplicate texts deduplicated"
            )

        texts_ds, removed = self._prepare_text_dataset(
            unique_texts, task.metadata, instruction_template
        )

        if removed:
            old_to_new = self._build_index_remap(len(unique_texts), removed)
            valid_mask = [
                indices1[i] not in removed and indices2[i] not in removed
                for i in range(len(indices1))
            ]
            indices1 = [old_to_new[indices1[i]] for i in range(len(indices1)) if valid_mask[i]]
            indices2 = [old_to_new[indices2[i]] for i in range(len(indices2)) if valid_mask[i]]
            normalized_scores = [normalized_scores[i] for i in range(len(normalized_scores)) if valid_mask[i]]
            if self.rank == 0:
                n_removed = sum(1 for v in valid_mask if not v)
                print(f"  {n_removed} STS pairs removed due to filtered texts")

        datasets[task_name] = {
            "dataset": {
                "texts": texts_ds,
                "indices1": indices1,
                "indices2": indices2,
                "labels": normalized_scores,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    def _prepare_summarization(
        self, task, task_name, eval_split, instruction_template, datasets
    ):
        data_split, hf_subset = abs_task_preprocessing(task, eval_split)

        text_col = task.text_column_name
        human_col = task.human_summaries_column_name
        machine_col = task.machine_summaries_column_name
        relevance_col = task.relevancy_column_name

        human_summaries = list(data_split[human_col])
        machine_summaries = list(data_split[machine_col])
        relevance = list(data_split[relevance_col])

        normalized_scores = [
            ((np.array(x) - task.min_score) / (task.max_score - task.min_score)).tolist()
            for x in relevance
        ]

        human_lens = [len(hs) for hs in human_summaries]
        machine_lens = [len(ms) for ms in machine_summaries]

        all_human = [s for hs in human_summaries for s in hs]
        all_machine = [s for ms in machine_summaries for s in ms]

        all_texts = all_human + all_machine
        unique_texts, text_to_idx = [], {}
        for text in all_texts:
            h = hash(text)
            if h not in text_to_idx:
                text_to_idx[h] = len(unique_texts)
                unique_texts.append(text)

        human_indices = [text_to_idx[hash(s)] for s in all_human]
        machine_indices = [text_to_idx[hash(s)] for s in all_machine]

        if self.rank == 0:
            n_dedup = len(all_texts) - len(unique_texts)
            print(
                f"  {n_dedup}/{len(all_texts)} duplicate summaries deduplicated"
            )
            print(
                f"  {len(human_summaries)} samples, "
                f"{sum(human_lens)} human summaries, "
                f"{sum(machine_lens)} machine summaries"
            )

        texts_ds, removed = self._prepare_text_dataset(
            unique_texts, task.metadata, instruction_template
        )

        if removed:
            old_to_new = self._build_index_remap(len(unique_texts), removed)
            new_human_indices = []
            new_machine_indices = []
            new_human_lens = []
            new_machine_lens = []
            new_gold_scores = []
            h_off, m_off = 0, 0
            n_samples_orig = len(human_lens)
            for i in range(n_samples_orig):
                h_len = human_lens[i]
                m_len = machine_lens[i]
                s_h = human_indices[h_off:h_off + h_len]
                s_m = machine_indices[m_off:m_off + m_len]
                s_scores = normalized_scores[i]
                kept_h = [idx for idx in s_h if idx not in removed]
                kept_m_scores = [
                    (idx, s_scores[j]) for j, idx in enumerate(s_m)
                    if idx not in removed
                ]
                if kept_h and kept_m_scores:
                    new_human_indices.extend(old_to_new[idx] for idx in kept_h)
                    new_machine_indices.extend(old_to_new[idx] for idx, _ in kept_m_scores)
                    new_human_lens.append(len(kept_h))
                    new_machine_lens.append(len(kept_m_scores))
                    new_gold_scores.append([s for _, s in kept_m_scores])
                h_off += h_len
                m_off += m_len
            human_indices = new_human_indices
            machine_indices = new_machine_indices
            human_lens = new_human_lens
            machine_lens = new_machine_lens
            normalized_scores = new_gold_scores
            if self.rank == 0 and len(human_lens) < n_samples_orig:
                print(
                    f"  {n_samples_orig - len(human_lens)} summarization samples "
                    f"removed due to filtered texts"
                )

        datasets[task_name] = {
            "dataset": {
                "texts": texts_ds,
                "human_indices": human_indices,
                "machine_indices": machine_indices,
                "human_lens": human_lens,
                "machine_lens": machine_lens,
                "gold_scores": normalized_scores,
            },
            "hf_split": eval_split,
            "main_score": task.metadata.main_score,
            "hf_subset": hf_subset,
            "task_type": task.metadata.type,
            "task_obj": task,
        }

    # -------------------------------------------------------------------------
    # Encoding helpers
    # -------------------------------------------------------------------------

    def _encode_dataset(self, model, dataset, batch_size):
        """Encode a prepared dataset using the DDP-aware pipeline."""
        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
            tokenizer=self.tokenizer,
            eot_id=self.tokenizer.eos_token_id if self.append_eos else None,
            add_special_tokens=self.add_special_tokens,
        )
        sampler = LenghtSortedSampler(dataset)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dist.barrier()
        start = time.time()
        embeddings = encode(
            model, loader, prompt_type=PromptType.query, world_size=self.world_size,
            pool_fn=self.pool_fn,
        )
        dist.barrier()
        # if self.rank == 0:
        #     print(f"  encoded {len(dataset)} samples in {time.time()-start:.2f}s")
        return embeddings


    # -------------------------------------------------------------------------
    # Per-task-type evaluation
    # -------------------------------------------------------------------------

    def evaluate_one(
        self,
        dataset,
        task_specific_scores,
        ignore_identical_ids,
        hf_split,
        hf_subset,
        main_score,
        model,
        batch_size=8,
        top_k=None,
        k_values=[1, 3, 5, 10, 20, 100, 1000],
        skip_first_result: bool = False,
    ):

        if top_k is None:
            top_k = max(k_values)
        model = model.eval()

        query_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["queries"]["id"])}
        doc_idx_to_id = {idx: id_ for idx, id_ in enumerate(dataset["corpus"]["id"])}


        collate_fn = partial(
            collate_fn_with_padding,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side=self.padding_side,
            tokenizer=self.tokenizer,
            eot_id=self.tokenizer.eos_token_id if self.append_eos else None,
            add_special_tokens=self.add_special_tokens,
        )

        sampler_queries = None
        if self.world_size > 1:
            sampler_queries = torch.utils.data.distributed.DistributedSampler(
                dataset["queries"], shuffle=False, drop_last=False
            )

        if self.new_inference_mode:
            sampler_queries = LenghtSortedSampler(dataset["queries"])

        # if self.rank == 0:
        #     print("num queries", len(dataset["queries"]))
        #     print("num documents", len(dataset["corpus"]))

        queries_loader = DataLoader(
            dataset["queries"],
            sampler=sampler_queries,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        dist.barrier()
        start = time.time()
        query_embeddings = encode(
            model,
            queries_loader,
            prompt_type=PromptType.query,
            world_size=self.world_size,
            pool_fn=self.pool_fn,
        )
        dist.barrier()



        chunk_size, query_chunk_size = estimate_chunk_sizes(query_embeddings)

        # Broadcast chunk sizes from rank 0 so the search loop has the
        # same number of iterations on every GPU (free_mem can differ).
        _cs = torch.tensor(
            [chunk_size, query_chunk_size], dtype=torch.long, device=f"cuda:{self.rank}"
        )
        dist.broadcast(_cs, src=0)
        chunk_size, query_chunk_size = int(_cs[0].item()), int(_cs[1].item())
        del _cs

        top_scores, top_indices = search(
            model=model,
            query_embeddings=query_embeddings,
            corpus_dataset=dataset["corpus"],
            collate_fn=collate_fn,
            top_k=top_k,
            batch_size=batch_size,
            estract_positives=False,
            chunk_size=chunk_size,
            pool_fn=self.pool_fn,
        )

        dist.barrier()

        top_scores = top_scores.cpu()
        top_indices = top_indices.tolist()

        results = {}
        for i in range(len(top_scores)):
            results[query_idx_to_id[i]] = {
                doc_idx_to_id[index]: top_scores[i, j].item()
                for j, index in enumerate(top_indices[i])
            }

        qrels = dataset["relevant_docs"]
        if ignore_identical_ids:
            # Remove identical ids from results dict in some datasets the queries are also in the documents so they must be removed.
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)

        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            cv_recall,
        ) = calculate_retrieval_scores(
            results,
            qrels,
            list(k_values),
            skip_first_result,
        )

        task_specific_scores_ = task_specific_scores(
            all_scores,
            dataset["relevant_docs"],
            results,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        _previous_results_model_meta = None
        scores = make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            cv_recall,
            task_specific_scores_,
            _previous_results_model_meta,
        )
        return {main_score: scores[main_score]}

    @torch.inference_mode()
    def evaluate_one_pair_classification(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        task_obj = task_data["task_obj"]
        main_score = task_data["main_score"]

        embeddings = self._encode_dataset(model, dataset["texts"], batch_size)
        embeddings_np = embeddings.cpu().numpy()

        emb1 = embeddings_np[dataset["indices1"]]
        emb2 = embeddings_np[dataset["indices2"]]

        cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
        manhattan_distances_ = paired_manhattan_distances(emb1, emb2)
        euclidean_distances_ = paired_euclidean_distances(emb1, emb2)
        dot_scores = np.sum(emb1 * emb2, axis=1)

        distances = PairClassificationDistances(
            cosine_scores=cosine_scores.tolist(),
            euclidean_distances=euclidean_distances_.tolist(),
            manhattan_distances=manhattan_distances_.tolist(),
            similarity_scores=cosine_scores.tolist(),
            dot_scores=dot_scores.tolist(),
        )

        scores = task_obj._compute_metrics(distances, dataset["labels"])

        # if self.rank == 0:
        #     for k, v in scores.items():
        #         if k.startswith("max_"):
        #             print(f"  {k}: {v:.4f}")

        return {main_score: scores[main_score]}

    @torch.inference_mode()
    def evaluate_one_multilabel_classification(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        task_obj = task_data["task_obj"]
        main_score = task_data["main_score"]

        # if self.rank == 0:
        #     print("  encoding train set...")
        train_embeddings = self._encode_dataset(
            model, dataset["train_texts"], batch_size
        )
        # if self.rank == 0:
        #     print("  encoding test set...")
        test_embeddings = self._encode_dataset(
            model, dataset["test_texts"], batch_size
        )

        X_train_all = train_embeddings.cpu().numpy()
        X_test = test_embeddings.cpu().numpy()

        train_labels = dataset["train_labels"]
        test_labels = dataset["test_labels"]

        binarizer = MultiLabelBinarizer()
        y_test = binarizer.fit_transform(test_labels)

        scores = []
        for i_exp in range(task_obj.n_experiments):
            sample_indices, _ = task_obj._undersample_data_indices(
                train_labels, task_obj.samples_per_label, None
            )
            X_train = X_train_all[sample_indices]
            y_train = binarizer.transform(
                [train_labels[idx] for idx in sample_indices]
            )

            y_pred, classifier = _evaluate_classifier(
                X_train, y_train, X_test, task_obj.evaluator
            )
            scores_exp = task_obj._calculate_scores(
                y_test, y_pred, X_test, classifier
            )
            scores.append(scores_exp)

            # if self.rank == 0:
            #     print(
            #         f"  experiment {i_exp + 1}/{task_obj.n_experiments}: "
            #         f"f1={scores_exp['f1']:.4f}, lrap={scores_exp['lrap']:.4f}"
            #     )

        avg_scores = {
            k: float(np.mean([s[k] for s in scores])) for k in scores[0]
        }
        # if self.rank == 0:
        #     print(f"  avg {main_score}: {avg_scores.get(main_score, 'N/A')}")

        return {main_score: avg_scores[main_score]}

    @torch.inference_mode()
    def evaluate_one_clustering(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        task_obj = task_data["task_obj"]
        main_score = task_data["main_score"]

        embeddings = self._encode_dataset(model, dataset["texts"], batch_size)
        embeddings_np = embeddings.cpu().numpy()

        labels = dataset["labels"]
        labels = [l if isinstance(l, list) else [l] for l in labels]

        v_measures, _ = _evaluate_clustering_bootstrapped(
            embeddings_np,
            labels,
            n_clusters=task_obj.n_clusters,
            cluster_size=task_obj.max_documents_per_cluster,
            kmean_batch_size=task_obj.k_mean_batch_size,
            max_depth=task_obj.max_depth,
            rng_state=task_obj.rng_state,
            seed=task_obj.seed,
        )

        all_v = list(itertools.chain.from_iterable(v_measures.values()))
        mean_v = float(np.mean(all_v))
        std_v = float(np.std(all_v))

        # if self.rank == 0:
        #     print(f"  v_measure={mean_v:.4f} (std={std_v:.4f})")

        return {main_score: mean_v}

    @torch.inference_mode()
    def evaluate_one_classification(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        task_obj = task_data["task_obj"]
        main_score = task_data["main_score"]

        if self.rank == 0:
            print("  encoding train set...")
        train_embeddings = self._encode_dataset(
            model, dataset["train_texts"], batch_size
        )
        if self.rank == 0:
            print("  encoding test set...")
        test_embeddings = self._encode_dataset(
            model, dataset["test_texts"], batch_size
        )

        X_train_all = train_embeddings.cpu().numpy()
        X_test = test_embeddings.cpu().numpy()

        train_labels = dataset["train_labels"]
        test_labels = dataset["test_labels"]

        evaluator_model = task_obj.evaluator_model
        if "random_state" in evaluator_model.get_params():
            evaluator_model = evaluator_model.set_params(
                random_state=task_obj.seed
            )

        scores = []
        idxs = None
        for i_exp in range(task_obj.n_experiments):
            if idxs is None:
                idxs = list(range(len(train_labels)))
            rng_state = np.random.RandomState(task_obj.seed + i_exp)
            rng_state.shuffle(idxs)

            label_counter = defaultdict(int)
            sampled_idxs = []
            for i in idxs:
                label = train_labels[i]
                if label_counter[label] < task_obj.samples_per_label:
                    sampled_idxs.append(i)
                    label_counter[label] += 1

            X_train = X_train_all[sampled_idxs]
            y_train = [train_labels[i] for i in sampled_idxs]

            clf = clone(evaluator_model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            scores_exp = task_obj._calculate_scores(test_labels, y_pred)
            scores.append(scores_exp)

            # if self.rank == 0:
            #     print(
            #         f"  experiment {i_exp + 1}/{task_obj.n_experiments}: "
            #         f"accuracy={scores_exp['accuracy']:.4f}, f1={scores_exp['f1']:.4f}"
            #     )

        avg_scores = {
            k: (
                float(np.mean(values))
                if (values := [s[k] for s in scores if s[k] is not None])
                else np.nan
            )
            for k in scores[0].keys()
        }
        # if self.rank == 0:
        #     print(f"  avg {main_score}: {avg_scores.get(main_score, 'N/A')}")

        return {main_score: avg_scores[main_score]}

    @torch.inference_mode()
    def evaluate_one_sts(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        task_obj = task_data["task_obj"]
        main_score = task_data["main_score"]

        embeddings = self._encode_dataset(model, dataset["texts"], batch_size)
        embeddings_np = embeddings.cpu().numpy()

        emb1 = embeddings_np[dataset["indices1"]]
        emb2 = embeddings_np[dataset["indices2"]]

        cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
        manhattan_distances_ = -paired_manhattan_distances(emb1, emb2)
        euclidean_distances_ = -paired_euclidean_distances(emb1, emb2)

        scores_dict = {
            "cosine_scores": cosine_scores.tolist(),
            "manhattan_distances": manhattan_distances_.tolist(),
            "euclidean_distances": euclidean_distances_.tolist(),
            "similarity_scores": None,
        }

        scores = task_obj._calculate_scores(scores_dict, dataset["labels"])

        # if self.rank == 0:
        #     for k, v in scores.items():
        #         print(f"  {k}: {v:.4f}")

        return {main_score: scores[main_score]}

    @torch.inference_mode()
    def evaluate_one_summarization(self, task_data, model, batch_size):
        model = model.eval()
        dataset = task_data["dataset"]
        main_score = task_data["main_score"]

        embeddings = self._encode_dataset(model, dataset["texts"], batch_size)
        embeddings_np = embeddings.cpu().numpy()

        human_indices = dataset["human_indices"]
        machine_indices = dataset["machine_indices"]
        human_lens = dataset["human_lens"]
        machine_lens = dataset["machine_lens"]
        gold_scores = dataset["gold_scores"]

        embs_human = embeddings_np[human_indices]
        embs_machine = embeddings_np[machine_indices]

        embs_human_per_sample = np.split(embs_human, np.cumsum(human_lens)[:-1])
        embs_machine_per_sample = np.split(embs_machine, np.cumsum(machine_lens)[:-1])

        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []

        n_skipped = 0
        for i, (embs_human_i, embs_machine_i) in enumerate(
            zip(embs_human_per_sample, embs_machine_per_sample)
        ):
            cosine_pred = []
            dot_pred = []
            human_scores = []

            embs_human_i_norm = embs_human_i / np.linalg.norm(
                embs_human_i, axis=1, keepdims=True
            )

            for emb_machine, gold_score in zip(embs_machine_i, gold_scores[i]):
                emb_machine_norm = emb_machine / np.linalg.norm(emb_machine)
                cos_scores = emb_machine_norm @ embs_human_i_norm.T
                dot_scores = emb_machine @ embs_human_i.T

                cosine_pred.append(float(np.max(cos_scores)))
                dot_pred.append(float(np.max(dot_scores)))
                human_scores.append(gold_score)

            if (
                len(set(human_scores)) == 1
                or len(set(dot_pred)) == 1
                or len(set(cosine_pred)) == 1
            ):
                n_skipped += 1
                continue

            cosine_spearman_scores.append(
                spearmanr(human_scores, cosine_pred).statistic
            )
            cosine_pearson_scores.append(
                pearsonr(human_scores, cosine_pred).statistic
            )
            dot_spearman_scores.append(
                spearmanr(human_scores, dot_pred).statistic
            )
            dot_pearson_scores.append(
                pearsonr(human_scores, dot_pred).statistic
            )

        scores = {
            "cosine_spearman": float(np.mean(cosine_spearman_scores)),
            "cosine_pearson": float(np.mean(cosine_pearson_scores)),
            "dot_spearman": float(np.mean(dot_spearman_scores)),
            "dot_pearson": float(np.mean(dot_pearson_scores)),
            "pearson": float(np.mean(cosine_pearson_scores)),
            "spearman": float(np.mean(cosine_spearman_scores)),
        }

        # if self.rank == 0:
        #     print(f"  {n_skipped} samples skipped (constant scores)")
        #     for k, v in scores.items():
        #         print(f"  {k}: {v:.4f}")

        return {main_score: scores[main_score]}

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_averages(results):
        """Compute per-type, micro, and macro averages from the results dict.

        Parameters
        ----------
        results : dict
            Output of ``evaluate``: mapping task_type ->
            [{task_name: (score_dict, duration)}].

        Returns
        -------
        dict with keys:
            - one key per task_type  -> average score over all tasks in that type
            - "micro_average"        -> average over every individual task
            - "macro_average"        -> average of the per-type averages
        """
        summary = {}
        all_scores = []
        type_averages = []

        for task_type, task_list in results.items():
            type_scores = []
            for task_dict in task_list:
                for _name, (score_dict, _duration) in task_dict.items():
                    score = list(score_dict.values())[0]
                    type_scores.append(score)
            type_avg = float(sum(type_scores) / len(type_scores))
            summary[task_type] = type_avg
            all_scores.extend(type_scores)
            type_averages.append(type_avg)

        summary["micro_average"] = float(sum(all_scores) / len(all_scores))
        summary["macro_average"] = float(sum(type_averages) / len(type_averages))
        return summary

    def evaluate(self, model, batch_size=64):
        results = defaultdict(list)
        
        n_tasks = len(self.datasets)
        for i, (name, task_data) in enumerate(self.datasets.items()):
            
            dist.barrier()
            start = time.time()
            task_type = task_data["task_type"]
            if self.rank ==0: 
                print(f"\nevaluating {name} task {i}/{n_tasks}")

            if task_type == "Retrieval":
                output_res = self.evaluate_one(
                    dataset=task_data["dataset"],
                    task_specific_scores=task_data["task_specific_scores"],
                    ignore_identical_ids=task_data["ignore_identical_ids"],
                    hf_split=task_data["hf_split"],
                    hf_subset=task_data["hf_subset"],
                    main_score=task_data["main_score"],
                    model=model,
                    batch_size=batch_size,
                )
            elif task_type == "PairClassification":
                output_res = self.evaluate_one_pair_classification(
                    task_data, model, batch_size
                )
            elif task_type == "MultilabelClassification":
                output_res = self.evaluate_one_multilabel_classification(
                    task_data, model, batch_size
                )
            elif task_type == "Clustering":
                output_res = self.evaluate_one_clustering(
                    task_data, model, batch_size
                )
            elif task_type == "Classification":
                output_res = self.evaluate_one_classification(
                    task_data, model, batch_size
                )
            elif task_type == "STS":
                output_res = self.evaluate_one_sts(
                    task_data, model, batch_size
                )
            elif task_type == "Summarization":
                output_res = self.evaluate_one_summarization(
                    task_data, model, batch_size
                )
            elif task_type == "Reranking":
                output_res = self.evaluate_one(
                    dataset=task_data["dataset"],
                    task_specific_scores=task_data["task_specific_scores"],
                    ignore_identical_ids=task_data["ignore_identical_ids"],
                    hf_split=task_data["hf_split"],
                    hf_subset=task_data["hf_subset"],
                    main_score=task_data["main_score"],
                    model=model,
                    batch_size=batch_size,
                )
            else:
                if self.rank == 0:
                    print(f"  skipping unsupported task type: {task_type}")
                continue

            dist.barrier()
            duration = time.time() - start
            if self.rank == 0:
                print(f"{name} evaluated in {duration/60:.2f} min")
            results[task_type].append({name: (output_res, duration)})

        summary = self.compute_averages(results)
        return results, summary
