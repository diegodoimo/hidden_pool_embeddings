from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
import torch
import numpy as np
import itertools
from inference.helpers import abs_task_preprocessing, iter_deferred_layers
from inference.evaluate.shared import (
    encode_dataset,
    make_collate_fn,
    prepare_text_dataset,
    EvalContext,
)

# Match MTEB: keep every subsampled row and truncate at tokenization (encode), instead of
# dropping rows whose prompts exceed a length budget in create_dataset.
CLUSTERING_TRUNCATION_MAX_LENGTH = 8192


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

        # Legacy (commented): same max_length filter as other tasks — removed long documents
        # entirely, unlike MTEB AbsTaskClustering._evaluate_subset, which keeps all
        # downsampled rows and relies on the encoder to truncate.
        # texts_ds, removed = prepare_text_dataset(
        #     sentences, task.metadata, instruction_template, tokenizer, rank
        # )

        texts_ds, removed = prepare_text_dataset(
            sentences,
            task.metadata,
            instruction_template,
            tokenizer,
            rank,
            max_length=CLUSTERING_TRUNCATION_MAX_LENGTH,
            skip_length_filter=True,
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

    # Legacy (commented): collate without truncation — long prompts only worked if rows
    # were pre-filtered in prepare_text_dataset.
    # collate_fn = make_collate_fn(
    #     eval_context.tokenizer,
    #     eval_context.padding_side,
    #     eval_context.eot_id,
    #     eval_context.add_special_tokens,
    # )

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
        truncation_max_length=CLUSTERING_TRUNCATION_MAX_LENGTH,
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


@torch.inference_mode()
def evaluate_hidden_states_clustering(
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

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
        truncation_max_length=CLUSTERING_TRUNCATION_MAX_LENGTH,
    )
    deferred = encode_dataset(
        model,
        dataset["texts"],
        batch_size,
        collate_fn,
        extract_hidden_repr=True,
        target_layers=target_layers,
        use_last_token=use_last_token,
        deferred_gather=True,
    )

    labels = dataset["labels"]
    labels = [l if isinstance(l, list) else [l] for l in labels]

    layer_performance = {}
    for layer, embeddings in iter_deferred_layers(
        deferred,
        target_layers=target_layers,
        layer_subset=layer_subset,
    ):
        embeddings_np = embeddings.cpu().numpy()
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
        layer_performance[layer] = float(np.mean(all_v))

    return layer_performance
