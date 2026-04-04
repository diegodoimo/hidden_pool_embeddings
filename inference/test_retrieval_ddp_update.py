import os
import torch.distributed as dist
from collections import defaultdict
import time

from utils.helpers import _print_ram

from inference.evaluate.eval_retrieval import (
    _prepare_retrieval,
    evaluate_one,
    evaluate_hidden_states_retrieval,
)
from inference.evaluate.eval_reranking import (
    _prepare_reranking,
    evaluate_one_reranking,
    evaluate_hidden_states_reranking,
)
from inference.evaluate.eval_pair_classification import (
    _prepare_pair_classification,
    evaluate_one_pair_classification,
    evaluate_hidden_states_pair_classification,
)
from inference.evaluate.eval_multilabel_classification import (
    _prepare_multilabel_classification,
    evaluate_one_multilabel_classification,
    evaluate_hidden_states_multilabel_classification,
)
from inference.evaluate.eval_classification import (
    _prepare_classification,
    evaluate_one_classification,
    evaluate_hidden_states_classification,
)
from inference.evaluate.eval_sts import (
    _prepare_sts,
    evaluate_one_sts,
    evaluate_hidden_states_sts,
)
from inference.evaluate.eval_summarization import (
    _prepare_summarization,
    evaluate_one_summarization,
    evaluate_hidden_states_summarization,
)
from inference.evaluate.eval_bitext_mining import (
    _prepare_bitext_mining,
    evaluate_one_bitext_mining,
    evaluate_hidden_states_bitext_mining,
)
from inference.evaluate.eval_clustering import (
    _prepare_clustering,
    evaluate_one_clustering,
    evaluate_hidden_states_clustering,
)
from inference.evaluate.shared import EvalContext
from models.modules import last_token_pool


def compute_layerwise_averages(results):
    """Compute per-layer summary statistics from hidden-states results.

    Parameters
    ----------
    results : dict
        Output of ``evaluate_hidden_states``: mapping task_type ->
        [{task_name: (layer_dict, duration)}], where layer_dict is
        {layer_name -> score}.

    Returns
    -------
    dict[str, list[float]]
        Keys are task types plus "micro_average" and "macro_average".
        Values are lists of scores, one per layer, in layer order.
    """
    # Collect ordered layer names from the first available task
    layer_names = None
    for task_list in results.values():
        for task_dict in task_list:
            for layer_dict, _ in task_dict.values():
                layer_names = list(layer_dict.keys())
                break
        if layer_names is not None:
            break
    if not layer_names:
        return {}

    summary = {
        key: [] for key in list(results.keys()) + ["micro_average", "macro_average"]
    }

    for layer in layer_names:
        all_scores = []
        type_averages = []
        for task_type, task_list in results.items():
            type_scores = [
                layer_dict[layer]
                for task_dict in task_list
                for layer_dict, _ in task_dict.values()
            ]
            type_avg = float(sum(type_scores) / len(type_scores))
            summary[task_type].append(type_avg)
            all_scores.extend(type_scores)
            type_averages.append(type_avg)
        summary["micro_average"].append(float(sum(all_scores) / len(all_scores)))
        summary["macro_average"].append(float(sum(type_averages) / len(type_averages)))

    return summary


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


class evaluate_retrieval:

    def __init__(
        self,
        tokenizer,
        tasks,
        instruction_template,
        padding_side="right",
        add_special_tokens=False,
        eot_id=None,
        max_samples=1_000_000,
        max_eval_queries=None,
    ):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.add_special_tokens = add_special_tokens
        self.eot_id = eot_id
        self.instruction_template = instruction_template
        self.max_samples = max_samples
        self.max_eval_queries = max_eval_queries

        t1 = time.time()
        _print_ram(label="before loading datasets", rank=self.rank)
        self.datasets = self.prepare_datasets(
            tasks=tasks,
            instruction_template=self.instruction_template,
            max_samples=self.max_samples,
            max_eval_queries=self.max_eval_queries,
        )
        dist.barrier()
        _print_ram(label="after loading datasets", rank=self.rank)
        if self.rank == 0:
            print(f"datasets prepared in {(time.time()-t1)/60:.2f} min")

    def update_datasets(self, tasks):
        del self.datasets
        self.datasets = self.prepare_datasets(
            tasks=tasks,
            instruction_template=self.instruction_template,
            max_samples=self.max_samples,
            max_eval_queries=self.max_eval_queries,
        )

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
        tasks,
        instruction_template,
        max_samples=1_000_000,
        max_eval_queries=None,
    ):

        datasets = {}
        n_tasks = len(tasks)
        for i, task in enumerate(tasks):
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
            # Hub metadata sums every config/split on the dataset card; for Reranking that
            # often massively overcounts unrelated configs, so we skip tasks like
            # MindSmallReranking while MTEB still evaluates them — then the Reranking
            # category average is wrong (e.g. only AskUbuntu ~0.63 vs golden ~0.47).
            skip_for_size = total_rows > max_samples and task_type != "Reranking"
            if skip_for_size:
                if self.rank == 0:
                    print(
                        f"  SKIPPING {task_name}: total across {num_configs} configs "
                        f"is {total_rows} rows (largest: {max_rows} in {size_info}, "
                        f"limit={max_samples})"
                    )
                continue

            task.load_data()
            if task_type == "Retrieval":
                _prepare_retrieval(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "PairClassification":
                _prepare_pair_classification(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "MultilabelClassification":
                _prepare_multilabel_classification(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "Clustering":
                _prepare_clustering(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "Classification":
                _prepare_classification(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "STS":
                _prepare_sts(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "Summarization":
                _prepare_summarization(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "BitextMining":
                _prepare_bitext_mining(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                )
            elif task_type == "Reranking":
                _prepare_reranking(
                    task,
                    task_name,
                    eval_split,
                    instruction_template,
                    datasets,
                    self.tokenizer,
                    self.rank,
                    max_eval_queries=max_eval_queries,
                )
            else:
                if self.rank == 0:
                    print(f"  WARNING: unsupported task type '{task_type}', skipping")
            # _print_ram(label="dataset loaded", rank=self.rank)

        return datasets

    def evaluate(self, model, batch_size=64):

        model.eval()

        results = defaultdict(list)
        eval_context = EvalContext(
            tokenizer=self.tokenizer,
            padding_side=self.padding_side,
            eot_id=self.eot_id,
            add_special_tokens=self.add_special_tokens,
            world_size=self.world_size,
            rank=self.rank,
        )

        n_tasks = len(self.datasets)
        for i, (name, task_data) in enumerate(self.datasets.items()):

            dist.barrier()
            start = time.time()
            task_type = task_data["task_type"]
            if self.rank == 0:
                print(f"\nevaluating {name} task {i}/{n_tasks}")

            if task_type == "Retrieval":
                output_res = evaluate_one(task_data, model, batch_size, eval_context)
            elif task_type == "PairClassification":
                output_res = evaluate_one_pair_classification(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "MultilabelClassification":
                output_res = evaluate_one_multilabel_classification(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "Clustering":
                output_res = evaluate_one_clustering(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "Classification":
                output_res = evaluate_one_classification(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "STS":
                output_res = evaluate_one_sts(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "Summarization":
                output_res = evaluate_one_summarization(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "BitextMining":
                output_res = evaluate_one_bitext_mining(
                    task_data, model, batch_size, eval_context
                )
            elif task_type == "Reranking":
                output_res = evaluate_one_reranking(
                    task_data, model, batch_size, eval_context
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

        summary = compute_averages(results)
        return results, summary

    def evaluate_hidden_states(
        self,
        model,
        batch_size=64,
        target_layers=None,
        use_last_token=None,
    ):
        """Evaluate hidden-state representations layer by layer.

        *use_last_token* is auto-detected from the model's pooling function
        (True for last-token poolers, False for mean poolers such as
        embeddinggemma).  Pass an explicit bool to override detection.
        """
        model.eval()

        if use_last_token is None:
            m = model.module if hasattr(model, "module") else model
            use_last_token = getattr(m, "pool_fn", None) is last_token_pool
        if self.rank == 0:
            print(f"hidden-states pooling: {'last_token' if use_last_token else 'mean'}")

        results = defaultdict(list)
        eval_context = EvalContext(
            tokenizer=self.tokenizer,
            padding_side=self.padding_side,
            eot_id=self.eot_id,
            add_special_tokens=self.add_special_tokens,
            world_size=self.world_size,
            rank=self.rank,
        )

        n_tasks = len(self.datasets)
        for i, (name, task_data) in enumerate(self.datasets.items()):

            dist.barrier()
            start = time.time()
            task_type = task_data["task_type"]
            if self.rank == 0:
                print(f"\nevaluating {name} task {i}/{n_tasks}")

            if task_type == "Retrieval":
                output_res = evaluate_hidden_states_retrieval(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "PairClassification":
                output_res = evaluate_hidden_states_pair_classification(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "MultilabelClassification":
                output_res = evaluate_hidden_states_multilabel_classification(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "Clustering":
                output_res = evaluate_hidden_states_clustering(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "Classification":
                output_res = evaluate_hidden_states_classification(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "STS":
                output_res = evaluate_hidden_states_sts(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "Summarization":
                output_res = evaluate_hidden_states_summarization(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "BitextMining":
                output_res = evaluate_hidden_states_bitext_mining(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
                )
            elif task_type == "Reranking":
                output_res = evaluate_hidden_states_reranking(
                    task_data,
                    model,
                    batch_size,
                    eval_context,
                    target_layers,
                    use_last_token,
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

        summary = compute_layerwise_averages(results)
        return results, summary
