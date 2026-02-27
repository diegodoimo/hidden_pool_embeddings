import os
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
from utils.dataloader_helpers import LenghtSortedSampler
from inference.helpers import (
    search,
    encode,
    abs_task_preprocessing,
)

from utils.dataloader_helpers import collate_fn_with_padding
from inference.hard_negative_mining import estimate_chunk_sizes
from utils.create_datasets import create_dataset
from utils.helpers import _print_ram
from collections import defaultdict
import time
from datasets import Dataset as HFDataset

from inference.evaluate.eval_clustering import (
    _prepare_clustering as _prepare_clustering_fn,
    evaluate_one_clustering as evaluate_one_clustering_fn,
)
from inference.evaluate.eval_retrieval import (
    _prepare_retrieval as _prepare_retrieval_fn,
    evaluate_one as evaluate_one_fn,
)
from inference.evaluate.eval_pair_classification import (
    _prepare_pair_classification as _prepare_pair_classification_fn,
    evaluate_one_pair_classification as evaluate_one_pair_classification_fn,
)
from inference.evaluate.eval_multilabel_classification import (
    _prepare_multilabel_classification as _prepare_multilabel_classification_fn,
    evaluate_one_multilabel_classification as evaluate_one_multilabel_classification_fn,
)
from inference.evaluate.eval_classification import (
    _prepare_classification as _prepare_classification_fn,
    evaluate_one_classification as evaluate_one_classification_fn,
)
from inference.evaluate.eval_sts import (
    _prepare_sts as _prepare_sts_fn,
    evaluate_one_sts as evaluate_one_sts_fn,
)
from inference.evaluate.eval_summarization import (
    _prepare_summarization as _prepare_summarization_fn,
    evaluate_one_summarization as evaluate_one_summarization_fn,
)
from inference.evaluate.eval_bitext_mining import (
    _prepare_bitext_mining as _prepare_bitext_mining_fn,
    evaluate_one_bitext_mining as evaluate_one_bitext_mining_fn,
)


class evaluate_retrieval:

    def __init__(
        self,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
        new_inference_mode=True,
        add_special_tokens=False,
        eot_id=None,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tokenizer = tokenizer
        # self.task_names = [task.metadata.name for task in tasks]
        self.tasks = tasks
        self.padding_side = padding_side
        self.new_inference_mode = new_inference_mode
        self.add_special_tokens = add_special_tokens
        self.eot_id = eot_id

        t1 = time.time()
        _print_ram(label="before loading datasets", rank=self.rank)
        self.datasets = self.prepare_datasets(instruction_template)
        dist.barrier()
        _print_ram(label="after loading datastes", rank=self.rank)
        if self.rank == 0:
            print(f"datsets prepared in {(time.time()-t1)/60:.2f} min")

    def _get_max_split_size_from_hub(self, task):
        """Query HF Hub for the largest split size (rows) without downloading data.

        Inspects all dataset configs (e.g. corpus/queries for retrieval tasks)
        and returns (max_rows, total_rows, num_configs, description_string).
        Only metadata is fetched.
        """
        from datasets import get_dataset_config_names, get_dataset_config_info

        ds_kwargs = dict(task.metadata.dataset)
        path = ds_kwargs.get("path")
        if not path:
            return 0, 0, 0, ""
        revision = ds_kwargs.get("revision")

        try:
            configs = get_dataset_config_names(path, revision=revision)
        except Exception as e:
            if self.rank == 0:
                print(f"  WARNING: could not list configs for {path}: {e}")
            return 0, 0, 0, ""

        max_rows = 0
        total_rows = 0
        max_info = ""
        for config in configs:
            try:
                info = get_dataset_config_info(
                    path, config_name=config, revision=revision
                )
                if info.splits:
                    for split_name, split_info in info.splits.items():
                        total_rows += split_info.num_examples
                        if split_info.num_examples > max_rows:
                            max_rows = split_info.num_examples
                            max_info = f"{config}/{split_name}"
            except Exception:
                continue

        return max_rows, total_rows, len(configs), max_info

    def prepare_datasets(
        self,
        instruction_template,
        max_samples=500_000,
    ):

        datasets = {}
        n_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            task_name = task.metadata.name
            task_type = task.metadata.type

            if self.rank == 0:
                print(f"processing {task_name} (type: {task_type}) {i}/{n_tasks}")

            eval_splits = task.metadata.eval_splits
            if "test" in eval_splits:
                eval_split = "test"
            else:
                eval_split = eval_splits[-1]
            if self.rank == 0 and len(eval_splits) > 1:
                print(f"  multiple eval_splits {eval_splits}, using '{eval_split}'")

            max_rows, total_rows, num_configs, size_info = (
                self._get_max_split_size_from_hub(task)
            )
            if self.rank == 0 and total_rows == 0:
                print(f"  (hub metadata unavailable, skipping size pre-check)")
            if total_rows > max_samples:
                if self.rank == 0:
                    print(
                        f"  SKIPPING {task_name}: total across {num_configs} configs "
                        f"is {total_rows} rows (largest: {max_rows} in {size_info}, "
                        f"limit={max_samples})"
                    )
                continue

            if self.rank == 0 and total_rows > 0:
                print(
                    f"  {num_configs} configs, {total_rows} total rows "
                    f"(largest: {max_rows} in {size_info})"
                )

            task.load_data()
            if task_type == "Retrieval":
                _prepare_retrieval_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "PairClassification":
                _prepare_pair_classification_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "MultilabelClassification":
                _prepare_multilabel_classification_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Clustering":
                _prepare_clustering_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Classification":
                _prepare_classification_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "STS":
                _prepare_sts_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Summarization":
                _prepare_summarization_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "BitextMining":
                _prepare_bitext_mining_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            elif task_type == "Reranking":
                _prepare_retrieval_fn(
                    self, task, task_name, eval_split, instruction_template, datasets
                )
            else:
                if self.rank == 0:
                    print(f"  WARNING: unsupported task type '{task_type}', skipping")
            _print_ram(label="dataset loaded", rank=self.rank)

        return datasets

    def _prepare_text_dataset(
        self, texts, task_metadata, instruction_template, max_length=8192
    ):
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
        removed_indices = (
            set(int(x) for x in ds.removed_ids) if ds.removed_ids else set()
        )
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
            eot_id=self.eot_id,
            add_special_tokens=self.add_special_tokens,
        )
        sampler = LenghtSortedSampler(dataset)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=max(1, len(os.sched_getaffinity(0)) // 2 - 2),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dist.barrier()
        embeddings = encode(
            model,
            loader,
            prompt_type=PromptType.query,
            world_size=self.world_size,
        )
        dist.barrier()
        return embeddings

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
            if self.rank == 0:
                print(f"\nevaluating {name} task {i}/{n_tasks}")

            if task_type == "Retrieval":
                output_res = evaluate_one_fn(
                    self,
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
                output_res = evaluate_one_pair_classification_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "MultilabelClassification":
                output_res = evaluate_one_multilabel_classification_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "Clustering":
                output_res = evaluate_one_clustering_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "Classification":
                output_res = evaluate_one_classification_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "STS":
                output_res = evaluate_one_sts_fn(self, task_data, model, batch_size)
            elif task_type == "Summarization":
                output_res = evaluate_one_summarization_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "BitextMining":
                output_res = evaluate_one_bitext_mining_fn(
                    self, task_data, model, batch_size
                )
            elif task_type == "Reranking":
                output_res = evaluate_one_fn(
                    self,
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
