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
from utils.sorted_sampler import LenghtSortedSampler
import time
import numpy as np
from utils.helpers import print_memory_consumed, return_formatted

# ***********************************************************************************************


def collate_fn_with_padding(
    batch, pad_token_id=0, padding_side="right", tokenizer=None, eot_id=None
):

    input_text = [item["prompt"] for item in batch]
    tokens = tokenizer(
        input_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    query_token_ids = [torch.tensor(tok + [eot_id]) for tok in tokens]

    # query_token_ids = [torch.tensor(item["input_ids"]) for item in batch]
    query_attention_mask = [torch.ones_like(input_ids) for input_ids in query_token_ids]

    # Pad queries and create attention masks
    query_token_ids_padded = pad_sequence(
        query_token_ids,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )

    query_attention_mask = pad_sequence(
        query_attention_mask,
        batch_first=True,
        padding_value=0,
        padding_side=padding_side,
    )

    assert query_token_ids_padded.dtype == torch.int64, batch
    return {
        "input_ids": query_token_ids_padded,
        "attention_mask": query_attention_mask,
    }


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def abs_task_preprocessing(task, eval_split):

    subsets_to_run = None
    task.dataset = cast(dict[HFSubset, DatasetDict], task.dataset)

    if task.hf_subsets is None:
        hf_subsets = list(task.dataset.keys())
    else:
        hf_subsets = copy(task.hf_subsets)

    if subsets_to_run is not None:  # allow overwrites of pre-filtering
        hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

    for hf_subset in hf_subsets:
        if hf_subset not in task.dataset and hf_subset == "default":
            data_split = task.dataset[eval_split]
        else:
            data_split = task.dataset[hf_subset][eval_split]
    assert len(hf_subsets) == 1, hf_subsets
    return data_split, hf_subset


@torch.inference_mode()
def search(
    model,
    query_embeddings,
    corpus_dataset,
    collate_fn,
    n_positives,
    qrels_query_ids,
    qrels_positive_ids,
    unique_query_ids,
    unique_query_id_to_idx,
    corpus_ids,
    top_k=100,
    batch_size=64,
    chunk_size=10**4,
    query_chunk_size=None,
    print_every=2*10**5,
):
    """
    Search for top-k documents and compute query-positive scores.

    When *query_chunk_size* is smaller than *N_queries*, the similarity
    computation is tiled over query sub-chunks so that the
    ``[query_chunk, corpus_chunk]`` score matrix fits in GPU memory.

    Supports both GPU-resident and CPU-resident *query_embeddings*.  When
    queries live on CPU, ``top_scores`` / ``top_indices`` are also kept on
    CPU and query slices are moved to GPU on-the-fly for each matmul.  The
    final distributed merge is performed in chunks through GPU to avoid OOM.

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
    queries_on_cpu = not query_embeddings.is_cuda
    gpu_device = torch.device(f"cuda:{rank}")

    if query_chunk_size is None:
        query_chunk_size = N_queries

    # top_scores / top_indices live on the same device as query_embeddings
    # (CPU for very large query sets, GPU otherwise)
    result_device = query_embeddings.device
    top_scores = torch.full((N_queries, top_k), -float("inf"), device=result_device)
    top_indices = torch.full(
        (N_queries, top_k), -1, dtype=torch.long, device=result_device
    )

    # ========================================================================
    # PREPARATION FOR QUERY-POSITIVE SCORE EXTRACTION
    # ========================================================================

    # Build mapping from corpus_id to corpus_index (only for positives)
    corpus_id_to_idx = {pid: idx for idx, pid in enumerate(corpus_ids[:n_positives])}

    # Build inverted index: positive_corpus_idx -> list of (query_idx, qrel_idx)
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
        if queries_on_cpu:
            print("Query embeddings on CPU — top-k bookkeeping also on CPU")
        if query_chunk_size < N_queries:
            print(
                f"Using query_chunk_size: {return_formatted(query_chunk_size)} "
                f"({(N_queries + query_chunk_size - 1) // query_chunk_size} query passes per corpus chunk)"
            )
        print(
            f"Will extract query-positive scores for {return_formatted(len(qrels_query_ids))} qrels pairs"
        )

    time_loading = 0
    time_encoding = 0
    time_sim = 0

    start = time.time()
    # print(f"chunk size {rank}: {chunk_size}")
    # #n_iters = N_corpus//chunk_size
    # print(f"iters {rank}: {n_iters}")

    for i, chunk_idx in enumerate(range(0, N_corpus, chunk_size)):

        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # t0 = time.time()

        if (i + 1) % interval == 0:
            #print(rank)
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
            num_workers=16,
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

            # Move query slice to GPU if needed (no-op when already on GPU)
            q_slice = query_embeddings[q_start:q_end]
            if queries_on_cpu:
                q_slice = q_slice.to(gpu_device)

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
        # torch.cuda.synchronize()
        # t11 = time.time()

        # time_loading = t01 - t0
        # time_encoding += t1 - t01
        # time_sim += t11 - t1


    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("\nGathering top scores top indices from all GPUs...")

    # print(
    #     f"{rank}: {top_scores.shape} {top_indices.shape} {query_positive_scores.shape}"
    # )

    # Distributed merging for top-k results
    if world_size > 1:
        if queries_on_cpu:
            # Chunked merge through GPU to avoid OOM
            merge_chunk = max(100_000, query_chunk_size)
            for q_start in range(0, N_queries, merge_chunk):
                q_end = min(q_start + merge_chunk, N_queries)

                ts = top_scores[q_start:q_end].to(gpu_device)
                ti = top_indices[q_start:q_end].to(gpu_device)

                scores_list = [torch.empty_like(ts) for _ in range(world_size)]
                indices_list = [torch.empty_like(ti) for _ in range(world_size)]
                dist.all_gather(scores_list, ts)
                dist.all_gather(indices_list, ti)

                all_scores = torch.cat(scores_list, dim=1)
                all_indices = torch.cat(indices_list, dim=1)

                merged_scores, merge_idx = torch.topk(
                    all_scores,
                    k=top_k,
                    dim=1,
                    largest=True,
                )
                merged_indices = torch.gather(all_indices, 1, merge_idx)

                top_scores[q_start:q_end] = merged_scores.cpu()
                top_indices[q_start:q_end] = merged_indices.cpu()

                del ts, ti, scores_list, indices_list
                del all_scores, all_indices, merged_scores, merge_idx, merged_indices
                torch.cuda.empty_cache()
        else:
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
    if rank == 0:
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

    if rank == 0:
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

    # Move chunk results to same device as top_scores (may be CPU)
    target_device = top_scores.device
    if chunk_top_scores.device != target_device:
        chunk_top_scores = chunk_top_scores.to(target_device)
        chunk_absolute_indices = chunk_absolute_indices.to(target_device)

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
    model, loader, world_size, prompt_type, divided_by_chunks=False, stream_to_cpu=False
):

    # distributed sampler will duplicate examples at the end
    indices = None
    if hasattr(loader.sampler, "indices"):
        indices = loader.sampler.indices
        assert isinstance(indices, list)

    num_samples = len(loader.dataset)
    embeddings = []

    # Use CUDA prefetcher to overlap H2D transfer with forward pass
    # prefetcher = CUDAPrefetcher(loader, device=model.device)

    for batch in loader:
        # batch is already on GPU (transferred asynchronously by the prefetcher)
        batch = {key: val.to(model.device) for key, val in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            out_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            out_embeddings = last_token_pool(
                out_embeddings.last_hidden_state,
                batch["attention_mask"],
            )
            batch_embeddings = F.normalize(out_embeddings, p=2, dim=1)
        if stream_to_cpu:
            embeddings.append(batch_embeddings.float().cpu())
        else:
            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    if not stream_to_cpu:
        embeddings = embeddings.float()
    # stream_to_cpu: already float32 and on CPU

    if stream_to_cpu:
        indices = torch.tensor(indices)  # keep on CPU
    else:
        indices = torch.tensor(indices, device=embeddings.device)

    if prompt_type == PromptType.document and divided_by_chunks:
        # if we are processing documents divided in chunks, we postpone the allgather
        # (in the distributed setup) and return the (local) indices
        return embeddings, indices

    if world_size > 1:
        if stream_to_cpu:
            # Chunked all-gather through GPU to avoid GPU OOM
            device = torch.device(f"cuda:{dist.get_rank()}")
            embeddings, indices = _chunked_all_gather_to_cpu(
                embeddings,
                indices,
                world_size,
                num_samples,
                device,
            )
        else:
            gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
            dist.all_gather(gathered, embeddings)
            # Concatenate across ranks for this batch
            embeddings = torch.cat(gathered, dim=0)
            embeddings = embeddings[:num_samples]

            # Also gather indices if we tracked them
            if indices is not None:
                gathered_indices = [
                    torch.zeros_like(indices) for _ in range(world_size)
                ]
                dist.all_gather(gathered_indices, indices)
                indices = torch.cat(gathered_indices, dim=0)
                indices = indices[:num_samples]

    # Restore original order
    sorted_positions = torch.argsort(indices)
    embeddings = embeddings[sorted_positions]
    return embeddings


def _chunked_all_gather_to_cpu(
    local_embeddings,
    local_indices,
    world_size,
    num_samples,
    device,
    chunk_size=500_000,
):
    """All-gather CPU tensors from all ranks via GPU in manageable chunks.

    Each chunk is moved to GPU, gathered with NCCL ``all_gather``, then
    moved back to CPU so that the full gathered result never resides on GPU
    at once.

    The ordering of rows in the returned tensors differs from a monolithic
    ``all_gather``, but ``(embedding, index)`` pairs are preserved so the
    caller can sort by index to recover the original order.
    """
    local_n = local_embeddings.shape[0]
    all_emb_chunks = []
    all_idx_chunks = []

    for start in range(0, local_n, chunk_size):
        end = min(start + chunk_size, local_n)

        emb_gpu = local_embeddings[start:end].to(device)
        idx_gpu = local_indices[start:end].to(device)

        gathered_emb = [torch.empty_like(emb_gpu) for _ in range(world_size)]
        gathered_idx = [torch.empty_like(idx_gpu) for _ in range(world_size)]
        dist.all_gather(gathered_emb, emb_gpu)
        dist.all_gather(gathered_idx, idx_gpu)

        all_emb_chunks.append(torch.cat([g.cpu() for g in gathered_emb], dim=0))
        all_idx_chunks.append(torch.cat([g.cpu() for g in gathered_idx], dim=0))

        del emb_gpu, idx_gpu, gathered_emb, gathered_idx
        torch.cuda.empty_cache()

    all_embeddings = torch.cat(all_emb_chunks, dim=0)[:num_samples]
    all_indices = torch.cat(all_idx_chunks, dim=0)[:num_samples]
    return all_embeddings, all_indices
