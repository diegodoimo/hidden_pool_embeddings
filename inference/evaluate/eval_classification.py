import torch
import numpy as np
from typing import cast
from copy import copy
from collections import defaultdict
from mteb.types import HFSubset
from datasets import DatasetDict
from sklearn.base import clone

from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)


def _prepare_classification(
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
def evaluate_one_classification(task_data, model, batch_size, eval_context: EvalContext):
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

    evaluator_model = task_obj.evaluator_model
    if "random_state" in evaluator_model.get_params():
        evaluator_model = evaluator_model.set_params(random_state=task_obj.seed)

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

    avg_scores = {
        k: (
            float(np.mean(values))
            if (values := [s[k] for s in scores if s[k] is not None])
            else np.nan
        )
        for k in scores[0].keys()
    }

    return {main_score: avg_scores[main_score]}
