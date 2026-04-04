import os
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import cast
from copy import copy
from mteb.types import HFSubset
from datasets import DatasetDict

from torch.utils.data import DataLoader
from mteb.types import PromptType

import torch.distributed as dist
import torch.nn.functional as F
from utils.dataloader_helpers import LenghtSortedSampler
import time
import numpy as np
from utils.helpers import print_memory_consumed, return_formatted

# ***********************************************************************************************


def estimate_chunk_sizes(query_embeddings, max_corpus_chunk=5 * 10**4):
    """Estimate corpus and query chunk sizes to stay within GPU memory.

    When the full query set fits comfortably, *query_chunk_size* equals
    *n_queries* (i.e. no query chunking).  For very large query sets the
    function picks a smaller *query_chunk_size* so that the similarity
    matrix ``[query_chunk, corpus_chunk]`` stays within the memory budget.

    Returns:
        (corpus_chunk_size, query_chunk_size)
    """
    free_mem, _ = torch.cuda.mem_get_info()
    n_queries = query_embeddings.shape[0]
    elem_size = query_embeddings.element_size()
    dim = query_embeddings.shape[1]

    bytes_per_doc = dim * elem_size  # one corpus embedding vector
    budget = int(0.8 * free_mem)

    bytes_per_sim_col = n_queries * elem_size  # one column of the full sim matrix

    # try without query chunking first
    corpus_chunk = budget // max(1, bytes_per_doc + bytes_per_sim_col)
    corpus_chunk = max(1000, min(int(corpus_chunk), max_corpus_chunk))

    total_needed = n_queries * corpus_chunk * elem_size + corpus_chunk * bytes_per_doc
    if total_needed <= budget:
        return corpus_chunk, n_queries

    # need query chunking
    corpus_chunk = min(max_corpus_chunk, 10**4)
    remaining = budget - corpus_chunk * bytes_per_doc
    query_chunk = remaining // max(1, corpus_chunk * elem_size)
    query_chunk = max(10**4, min(int(query_chunk), n_queries))

    return corpus_chunk, query_chunk


def abs_task_preprocessing(task, eval_split):
    """Return a list of (data_split, hf_subset) tuples – one per subset."""

    subsets_to_run = None
    task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)

    if task.hf_subsets is None:
        hf_subsets = list(task.dataset.keys())
    else:
        hf_subsets = copy(task.hf_subsets)

    if subsets_to_run is not None:
        hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

    result = []
    for hf_subset in hf_subsets:
        if hf_subset not in task.dataset and hf_subset == "default":
            data_split = task.dataset[eval_split]
        else:
            data_split = task.dataset[hf_subset][eval_split]
        result.append((data_split, hf_subset))
    return result


@torch.inference_mode()
def search(
    model,
    query_embeddings,
    corpus_dataset,
    collate_fn,
    n_positives=0,
    qrels_query_ids=None,
    qrels_positive_ids=None,
    unique_query_ids=None,
    unique_query_id_to_idx=None,
    corpus_ids=None,
    estract_positives=True,
    top_k=100,
    batch_size=64,
    chunk_size=10**4,
    query_chunk_size=None,
    print_every=2 * 10**5,
    verbose=False,
):
    """
    Search for top-k documents and compute query-positive scores.

    When *query_chunk_size* is smaller than *N_queries*, the similarity
    computation is tiled over query sub-chunks so that the
    ``[query_chunk, corpus_chunk]`` score matrix fits in GPU memory.

    Args:
        model: The embedding model
        query_embeddings: Pre-computed query embeddings [N_queries, embedding_dim]
        corpus_dataset: The corpus dataset to search
        collate_fn: Collation function for batching
        n_positives: Number of positive documents at the start of corpus
        qrels_query_ids: List of query IDs from qrels (with repetitions)
        qrels_positive_ids: List of positive IDs from qrels (with repetitions)
        unique_query_ids: List of unique query IDs
        unique_query_id_to_idx: Dict mapping query ID to index in query_embeddings
        corpus_ids: List of corpus document IDs
        top_k: Number of top documents to retrieve
        batch_size: Batch size for encoding
        chunk_size: Chunk size for corpus processing
        query_chunk_size: Query sub-chunk size (None = no query chunking)
        print_every: Print progress every N documents

    Returns:
        tuple: (top_scores, top_indices, query_positive_scores)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    N_queries = query_embeddings.shape[0]
    N_corpus = len(corpus_dataset)
    gpu_device = torch.device(f"cuda:{rank}")

    if query_chunk_size is None:
        query_chunk_size = N_queries

    top_scores = torch.full((N_queries, top_k), -float("inf"), device=gpu_device)
    top_indices = torch.full(
        (N_queries, top_k), -1, dtype=torch.long, device=gpu_device
    )

    # ========================================================================
    # PREPARATION FOR QUERY-POSITIVE SCORE EXTRACTION
    # ========================================================================

    # Build mapping from corpus_id to corpus_index (only for positives)

    # Build inverted index: positive_corpus_idx -> list of (query_idx, qrel_idx)
    if estract_positives:
        assert n_positives > 0
        assert qrels_query_ids is not None
        assert qrels_positive_ids is not None
        assert unique_query_ids is not None
        assert unique_query_id_to_idx is not None
        assert corpus_ids is not None

        corpus_id_to_idx = {
            pid: idx for idx, pid in enumerate(corpus_ids[:n_positives])
        }
        positive_to_queries = {}
        for qrel_idx, (qid, pid) in enumerate(zip(qrels_query_ids, qrels_positive_ids)):
            corpus_idx = corpus_id_to_idx.get(pid)
            if corpus_idx is not None and corpus_idx < n_positives:
                query_idx = unique_query_id_to_idx[qid]
                if corpus_idx not in positive_to_queries:
                    positive_to_queries[corpus_idx] = []
                positive_to_queries[corpus_idx].append((query_idx, qrel_idx))

        # query_positive_scores always on GPU (one float per qrel pair — small)
        query_positive_scores = torch.zeros(len(qrels_query_ids), device=gpu_device)

    # ========================================================================

    interval = print_every // chunk_size + 1
    dist.barrier()
    start = time.time()
    if rank == 0:
        print(f"Using chunk_size: {return_formatted(chunk_size)}")
        if query_chunk_size < N_queries:
            print(
                f"Using query_chunk_size: {return_formatted(query_chunk_size)} "
                f"({(N_queries + query_chunk_size - 1) // query_chunk_size} query passes per corpus chunk)"
            )
        if estract_positives:
            print(
                f"Will extract query-positive scores for {return_formatted(len(qrels_query_ids))} qrels pairs"
            )

    time_loading = 0
    time_encoding = 0
    time_sim = 0
    start = time.time()

    for i, chunk_idx in enumerate(range(0, N_corpus, chunk_size)):

        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # t0 = time.time()

        if (i + 1) % interval == 0:
            # print(rank)
            if rank == 0:
                print(
                    f"processed {return_formatted(chunk_idx)}/{return_formatted(N_corpus)} samples in {(time.time()-start)/60:.2f} mins"
                )
                print_memory_consumed(rank=rank)
            # print(
            #     f"Time loading: {time_loading/60:.2f}min, Time encoding: {time_encoding/60:.2f}min, Time sim+topk: {time_sim/60:.2f}min, Total: {(time_loading+time_encoding+time_sim)/60:.2f}min"
            # )

        # Compute embeddings on-the-fly
        subcorpus = corpus_dataset.select(
            range(chunk_idx, min(chunk_idx + chunk_size, N_corpus))
        )

        sampler_corpus = LenghtSortedSampler(subcorpus)
        corpus_loader = DataLoader(
            subcorpus,
            sampler=sampler_corpus,
            batch_size=batch_size,
            num_workers=max(1, len(os.sched_getaffinity(0)) // 2 - 2),
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=False,
            collate_fn=collate_fn,
        )

        # torch.cuda.synchronize()
        # t01 = time.time()

        local_corpus_chunk, local_indices = encode(
            model,
            corpus_loader,
            prompt_type=PromptType.document,
            world_size=world_size,
            divided_by_chunks=True,
        )

        # torch.cuda.synchronize()
        # t1 = time.time()

        # Compute global indices for this chunk
        global_indices = local_indices + chunk_idx

        # Process similarity in query sub-chunks to bound memory
        for q_start in range(0, N_queries, query_chunk_size):
            q_end = min(q_start + query_chunk_size, N_queries)

            q_slice = query_embeddings[q_start:q_end]
            scores = torch.matmul(q_slice, local_corpus_chunk.T)
            del q_slice

            if chunk_idx < n_positives:
                query_positive_scores = update_query_positive_score(
                    query_positive_scores,
                    global_indices,
                    positive_to_queries,
                    scores,
                    q_start=q_start,
                    q_end=q_end,
                )

            (
                top_scores[q_start:q_end],
                top_indices[q_start:q_end],
            ) = update_hard_negatives(
                scores,
                local_indices,
                chunk_idx,
                top_scores[q_start:q_end],
                top_indices[q_start:q_end],
                top_k,
            )
            del scores

        del local_corpus_chunk, local_indices

    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0 and verbose:
        print("\nGathering top scores top indices from all GPUs...")

    # Distributed merging for top-k results
    if world_size > 1:
        scores_list = [torch.empty_like(top_scores) for _ in range(world_size)]
        indices_list = [torch.empty_like(top_indices) for _ in range(world_size)]

        dist.all_gather(scores_list, top_scores)
        dist.all_gather(indices_list, top_indices)

        all_scores = torch.cat(scores_list, dim=1)
        all_indices = torch.cat(indices_list, dim=1)

        top_scores, top_indices = torch.topk(
            all_scores,
            k=top_k,
            dim=1,
            largest=True,
        )
        top_indices = torch.gather(all_indices, 1, top_indices)

    dist.barrier()
    torch.cuda.synchronize()
    # Gather query-positive scores from all GPUs

    if not estract_positives:
        return top_scores, top_indices

    if rank == 0 and verbose:
        print("\nGathering query-positive scores from all GPUs...")
    if world_size > 1:
        # Each GPU has computed scores for some qrels pairs based on the chunks it processed
        # We need to sum the scores across GPUs (since each pair is computed by one GPU)
        gathered_scores = [
            torch.zeros_like(query_positive_scores) for _ in range(world_size)
        ]
        dist.all_gather(gathered_scores, query_positive_scores)

        # Sum across GPUs to get the final scores (only one GPU will have non-zero for each pair)
        query_positive_scores = torch.stack(gathered_scores).sum(dim=0)

    # Convert to numpy
    query_positive_scores = query_positive_scores.cpu().numpy()

    if rank == 0 and verbose and qrels_query_ids is not None:
        print(f"Query-positive scores computed for {len(qrels_query_ids)} pairs")

    return top_scores, top_indices, query_positive_scores


def update_hard_negatives(
    scores, local_indices, chunk_idx, top_scores, top_indices, top_k
):
    # topk on GPU (scores is always on GPU)
    chunk_top_scores, chunk_top_indices = torch.topk(
        scores,
        k=min(top_k, scores.shape[1]),
        dim=1,
        largest=True,
    )

    chunk_absolute_indices = local_indices[chunk_top_indices] + chunk_idx

    combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
    combined_indices = torch.cat([top_indices, chunk_absolute_indices], dim=1)
    del chunk_top_scores, chunk_absolute_indices

    # Find the true global top-k among the combined results
    top_k_in_combined_scores, top_k_in_combined_indices = torch.topk(
        combined_scores,
        k=top_k,
        dim=1,
        largest=True,
    )
    top_scores = top_k_in_combined_scores
    top_indices = torch.gather(combined_indices, 1, top_k_in_combined_indices)
    del combined_scores, combined_indices, top_k_in_combined_indices

    return top_scores, top_indices


def update_query_positive_score(
    query_positive_scores,
    global_indices,
    positive_to_queries,
    scores,
    q_start=0,
    q_end=None,
):
    """Extract query-positive scores from *scores*.

    When query chunking is active, *scores* has shape
    ``[q_end - q_start, corpus_chunk]`` and rows correspond to **global**
    query indices ``q_start … q_end-1``.  Only entries whose global query
    index falls within ``[q_start, q_end)`` are extracted.
    """
    # Convert global indices to set for faster lookup
    global_indices_set = set(global_indices.cpu().numpy())

    batch_queries = []
    batch_locals = []
    batch_qrels = []

    global_to_local = {g.item(): i for i, g in enumerate(global_indices)}

    for global_idx in global_indices_set:
        if global_idx in positive_to_queries:
            local_idx = global_to_local[global_idx]
            for query_idx, qrel_idx in positive_to_queries[global_idx]:
                # When chunking queries, skip entries outside [q_start, q_end)
                if q_end is not None and not (q_start <= query_idx < q_end):
                    continue
                batch_queries.append(query_idx - q_start)
                batch_locals.append(local_idx)
                batch_qrels.append(qrel_idx)

    if batch_queries:
        extracted_scores = scores[batch_queries, batch_locals]
        query_positive_scores[batch_qrels] = extracted_scores

    return query_positive_scores


@torch.inference_mode()
def encode(
    model,
    loader,
    world_size,
    prompt_type,
    divided_by_chunks=False,
):

    # distributed sampler will duplicate examples at the end
    indices = None
    if hasattr(loader.sampler, "indices"):
        indices = loader.sampler.indices
        assert isinstance(indices, list)

    num_samples = len(loader.dataset)
    embeddings = []

    for batch in loader:
        batch = {key: val.to(model.device) for key, val in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0).float()
    indices = torch.tensor(indices, device=embeddings.device)

    if prompt_type == PromptType.document and divided_by_chunks:
        # if we are processing documents divided in chunks, we postpone the allgather
        # (in the distributed setup) and return the (local) indices
        return embeddings, indices

    if world_size > 1:
        gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
        dist.all_gather(gathered, embeddings)
        all_embeddings = torch.cat(gathered, dim=0)

        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        dist.all_gather(gathered_indices, indices)
        all_indices = torch.cat(gathered_indices, dim=0)

        # Scatter into correct positions using the original indices.
        # Duplicates (from padding) harmlessly overwrite with the same value;
        # every valid index is guaranteed to be placed correctly.
        embeddings = torch.zeros(
            num_samples,
            all_embeddings.shape[1],
            device=all_embeddings.device,
            dtype=all_embeddings.dtype,
        )
        embeddings[all_indices] = all_embeddings
        return embeddings

    # Single-GPU: just restore original order
    sorted_positions = torch.argsort(indices)
    embeddings = embeddings[sorted_positions]
    return embeddings


@torch.inference_mode()
def encode_hidden_layers(
    model,
    loader,
    target_layers,
    world_size,
    use_last_token=False,
):
    """Extract and pool intermediate layer representations.

    Pooling mirrors the final embedding layer:
      - use_last_token=True  → last non-padding token
      - use_last_token=False → mean over non-padding tokens (average pooling)

    all_gather logic mirrors ``encode``: each rank processes its local shard,
    then results are gathered once at the end and reordered via sampler indices.
    This supports batch_size > 1 and avoids per-batch communication overhead.

    Returns:
        dict[str, Tensor]  layer_name -> Tensor[N_samples, hidden_dim]  (CPU)
    """
    indices = None
    if hasattr(loader.sampler, "indices"):
        indices = loader.sampler.indices
        assert isinstance(indices, list)

    num_samples = len(loader.dataset)

    # Temporary storage for the current batch's hook outputs
    hidden_states_tmp = {}

    def make_hook(name):
        def hook_fn(_module, _input, output):
            # Some layers return tuples (e.g. attention layers); take hidden states
            if isinstance(output, tuple):
                output = output[0]
            hidden_states_tmp[name] = output

        return hook_fn

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if name in target_layers
    ]

    # Accumulate pooled embeddings per layer across all local batches
    layer_embeddings = {name: [] for name in target_layers}

    try:
        for batch in loader:
            batch = {key: val.to(model.device) for key, val in batch.items()}
            attention_mask = batch["attention_mask"]  # [B, T]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=attention_mask,
                )

            seq_len = attention_mask.sum(dim=1)  # [B]

            for name in target_layers:
                activations = hidden_states_tmp[name]  # [B, T, D]

                if use_last_token:
                    B = seq_len.shape[0]
                    pooled = activations[
                        torch.arange(B, device=activations.device), seq_len - 1
                    ]  # [B, D]
                else:
                    mask = attention_mask.unsqueeze(-1).to(
                        activations.dtype
                    )  # [B, T, 1]
                    denom = mask.sum(dim=1)  # [B, 1]
                    pooled = (activations * mask).sum(dim=1) / denom  # [B, D]

                layer_embeddings[name].append(pooled.float().cpu())
    finally:
        for h in handles:
            h.remove()

    result = {}
    for name in target_layers:
        embeddings = torch.cat(layer_embeddings[name], dim=0)  # [local_N, D]

        if world_size > 1:
            embeddings = embeddings.to(model.device)
            indices_tensor = torch.tensor(indices, device=model.device)

            gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
            dist.all_gather(gathered, embeddings)
            all_embeddings = torch.cat(gathered, dim=0)

            gathered_indices = [
                torch.zeros_like(indices_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_indices, indices_tensor)
            all_indices = torch.cat(gathered_indices, dim=0)

            # Scatter into correct positions using the original indices.
            # Duplicates (from DistributedSampler padding) harmlessly overwrite
            # with the same value; every valid index is placed correctly.
            out = torch.zeros(
                num_samples, all_embeddings.shape[1], dtype=all_embeddings.dtype
            )
            out[all_indices.cpu()] = all_embeddings.cpu()
            result[name] = out
        else:
            if indices is not None:
                sorted_positions = torch.argsort(torch.tensor(indices))
                embeddings = embeddings[sorted_positions]
            result[name] = embeddings

    return result
