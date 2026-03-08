from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
import torch
import numpy as np
import itertools
from inference.helpers import abs_task_preprocessing
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)


def _prepare_clustering(
    task, task_name, eval_split, instruction_template, datasets, tokenizer, rank
):
    subset_list = abs_task_preprocessing(task, eval_split)

    for data_split, hf_subset in subset_list:
        input_col = task.input_column_name
        label_col = task.label_column_name

        sentences = list(data_split[input_col])
        labels = list(data_split[label_col])

        max_doc = getattr(task, "max_document_to_embed", None)
        max_frac = getattr(task, "max_fraction_of_documents_to_embed", None)

        if max_doc is not None and max_frac is not None:
            raise ValueError(
                "Both max_document_to_embed and max_fraction_of_documents_to_embed are set"
            )

        if max_frac is not None:
            max_docs = min(
                len(sentences),
                int(max_frac * len(sentences)),
            )
            indices = task.rng_state.sample(range(len(sentences)), k=max_docs)
            sentences = [sentences[i] for i in indices]
            labels = [labels[i] for i in indices]
        elif max_doc is not None:
            max_docs = min(len(sentences), max_doc)
            indices = task.rng_state.sample(range(len(sentences)), k=max_docs)
            sentences = [sentences[i] for i in indices]
            labels = [labels[i] for i in indices]

        if rank == 0:
            print(f"  [{hf_subset}] {len(sentences)} samples for clustering")

        texts_ds, removed = prepare_text_dataset(
            sentences, task.metadata, instruction_template, tokenizer, rank
        )

        if removed:
            labels = [l for i, l in enumerate(labels) if i not in removed]

        entry_name = f"{task_name}/{hf_subset}" if len(subset_list) > 1 else task_name
        datasets[entry_name] = {
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


@torch.inference_mode()
def evaluate_one_clustering(task_data, model, batch_size, eval_context: EvalContext):
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
    embeddings = encode_dataset(model, dataset["texts"], batch_size, collate_fn)
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

    return {main_score: mean_v}
