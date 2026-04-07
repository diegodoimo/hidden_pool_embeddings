import warnings
import torch
import numpy as np
from typing import cast
from copy import copy
from mteb.types import HFSubset
from datasets import DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from mteb.abstasks.multilabel_classification import _evaluate_classifier

from inference.helpers import iter_deferred_layers
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)

import time


def _prepare_multilabel_classification(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)
    hf_subsets = copy(task.hf_subsets) if task.hf_subsets else list(task.dataset.keys())

    for hf_subset in hf_subsets:
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

        try:
            if len(test_data) > 2000:
                split_ds = test_data.train_test_split(
                    test_size=2000, seed=42, stratify_by_column=label_col
                )
                test_data = split_ds["test"]
        except ValueError:
            if rank == 0:
                print(
                    f"  [{hf_subset}] Could not stratify test subsample, using full test set"
                )

        train_texts = list(train_data[input_col])
        test_texts = list(test_data[input_col])
        train_labels = list(train_data[label_col])
        test_labels = list(test_data[label_col])

        if rank == 0:
            print(
                f"  [{hf_subset}] train: {len(train_texts)}, test: {len(test_texts)} samples"
            )

        train_ds, train_removed = prepare_text_dataset(
            train_texts, task.metadata, instruction_template, tokenizer, rank
        )
        test_ds, test_removed = prepare_text_dataset(
            test_texts, task.metadata, instruction_template, tokenizer, rank
        )

        if train_removed:
            train_labels = [
                l for i, l in enumerate(train_labels) if i not in train_removed
            ]
        if test_removed:
            test_labels = [
                l for i, l in enumerate(test_labels) if i not in test_removed
            ]

        entry_name = f"{task_name}/{hf_subset}" if len(hf_subsets) > 1 else task_name
        datasets[entry_name] = {
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


@torch.inference_mode()
def evaluate_one_multilabel_classification(
    task_data, model, batch_size, eval_context: EvalContext
):
    model.eval()
    dataset = task_data["dataset"]
    task_obj = task_data["task_obj"]
    main_score = task_data["main_score"]

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    train_embeddings = encode_dataset(
        model, dataset["train_texts"], batch_size, collate_fn
    )
    test_embeddings = encode_dataset(
        model, dataset["test_texts"], batch_size, collate_fn
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
        y_train = binarizer.transform([train_labels[idx] for idx in sample_indices])

        with warnings.catch_warnings():
            if eval_context.rank != 0:
                warnings.simplefilter("ignore", ConvergenceWarning)
                
            y_pred, classifier = _evaluate_classifier(
                X_train, y_train, X_test, task_obj.evaluator
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            scores_exp = task_obj._calculate_scores(y_test, y_pred, X_test, classifier)
        scores.append(scores_exp)

    avg_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}

    return {main_score: avg_scores[main_score]}


@torch.inference_mode()
def evaluate_hidden_states_multilabel_classification(
    task_data,
    model,
    batch_size,
    eval_context: EvalContext,
    target_layers,
    use_last_token=False,
    layer_subset=None,
):
    model.eval()
    dataset = task_data["dataset"]
    task_obj = task_data["task_obj"]
    main_score = task_data["main_score"]


    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )

    #start = time.time()
    deferred_train = encode_dataset(
        model,
        dataset["train_texts"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
        deferred_gather=True,
    )
    deferred_test = encode_dataset(
        model,
        dataset["test_texts"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
        deferred_gather=True,
    )

    # end = time.time()
    # if eval_context.rank ==0:
    #     print(f"encoding lasted {(end-start)/60}")
    # start = time.time()


    train_labels = dataset["train_labels"]
    test_labels = dataset["test_labels"]

    binarizer = MultiLabelBinarizer()
    y_test = binarizer.fit_transform(test_labels)

    layer_performance = {}
    for layer, train_embs, test_embs in iter_deferred_layers(
        deferred_train, deferred_test,
        target_layers=target_layers,
        layer_subset=layer_subset,
    ):
        x_train_all = train_embs.numpy()
        x_test = test_embs.numpy()

        scores = []
        for _ in range(task_obj.n_experiments):
            sample_indices, _ = task_obj._undersample_data_indices(
                train_labels, task_obj.samples_per_label, None
            )
            x_train = x_train_all[sample_indices]
            y_train = binarizer.transform([train_labels[idx] for idx in sample_indices])
            with warnings.catch_warnings():
                if eval_context.rank != 0:
                    warnings.simplefilter("ignore", ConvergenceWarning)
                y_pred, classifier = _evaluate_classifier(
                    x_train, y_train, x_test, task_obj.evaluator
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                scores_exp = task_obj._calculate_scores(y_test, y_pred, x_test, classifier)
            scores.append(scores_exp)

        avg_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
        layer_performance[layer] = avg_scores[main_score]

    # end = time.time()
    # if eval_context.rank ==0:
    #     print(f"perfomrnce measurement lasted {(end-start)/60}")




    return layer_performance
