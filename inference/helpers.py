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
from utils.helpers import print_memory_consumed

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
    print_every=10**5,
):
    """
    Search for top-k documents and compute query-positive scores.

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
        print_every: Print progress every N documents

    Returns:
        tuple: (top_scores, top_indices, query_positive_scores)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    N_queries = query_embeddings.shape[0]
    N_corpus = len(corpus_dataset)

    top_scores = torch.full(
        (N_queries, top_k), -float("inf"), device=query_embeddings.device
    )
    top_indices = torch.full(
        (N_queries, top_k), -1, dtype=torch.long, device=query_embeddings.device
    )

    # ========================================================================
    # SIMPLIFIED PREPARATION FOR QUERY-POSITIVE SCORE EXTRACTION
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

    # Initialize query-positive scores array
    query_positive_scores = torch.zeros(
        len(qrels_query_ids), device=query_embeddings.device
    )

    # ========================================================================

    interval = print_every // chunk_size + 1
    dist.barrier()
    start = time.time()
    if rank == 0:
        print(f"Using chunk_size: {chunk_size}")
        print(
            f"Will extract query-positive scores for {len(qrels_query_ids)} qrels pairs"
        )

    time_before = 0
    time_pos = 0
    time_hard = 0

    for i, chunk_idx in enumerate(range(0, N_corpus, chunk_size)):

        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        t0 = time.time()

        if (i + 1) % interval == 0 and rank == 0:
            # if rank == 0:
            print(
                f"processed {chunk_idx//10**3}k/{N_corpus//10**3}k samples in {(time.time()-start)/60} mins"
            )
            print(
                f"Time before: {time_before/60:.2f}min, Time pos: {time_pos/60:.2f}min, Time hard: {time_hard/60:.2f}min, Total: {(time_before+time_pos+time_hard)/60:.2f}min"
            )
            print_memory_consumed(rank=rank)

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
            collate_fn=collate_fn,
        )

        local_corpus_chunk, local_indices = encode(
            model,
            corpus_loader,
            prompt_type=PromptType.document,
            world_size=world_size,
            divided_by_chunks=True,
        )

        # Compute global indices for this chunk
        global_indices = local_indices + chunk_idx

        # Compute similarity scores for all query-document pairs in this chunk
        scores = torch.matmul(query_embeddings, local_corpus_chunk.T)
        del local_corpus_chunk  # Free corpus embeddings immediately

        torch.cuda.synchronize()
        t1 = time.time()

        if chunk_idx < n_positives:
            query_positive_scores = update_query_positive_score(
                query_positive_scores, global_indices, positive_to_queries, scores
            )

        torch.cuda.synchronize()
        t2 = time.time()

        top_scores, top_indices = update_hard_negatives(
            scores, local_indices, chunk_idx, top_scores, top_indices, top_k
        )

        torch.cuda.synchronize()
        t3 = time.time()

        time_before += t1 - t0
        time_pos += t2 - t1
        time_hard += t3 - t2

        del scores, local_indices

    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        print("\nFinal Benchmarking Results:")
        print(f"Total time before: {time_before/60:.2f}min")
        print(f"Total time update_query_positive_score: {time_pos/60:.2f}min")
        print(f"Total time update_hard_negatives: {time_hard/60:.2f}min")
        print(f"Total benchmarked time: {(time_before+time_pos+time_hard)/60:.2f}min")
        print(f"Total elapsed time: {(time.time()-start)/60:.2f}min\n")

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

    # Gather query-positive scores from all GPUs
    if rank == 0:
        print(f"\nGathering query-positive scores from all GPUs...")

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
    chunk_top_scores, chunk_top_indices = torch.topk(
        scores,
        k=min(top_k, scores.shape[1]),
        dim=1,
        largest=True,
    )
    # del scores  # Free the large scores tensor

    chunk_absolute_indices = local_indices[chunk_top_indices] + chunk_idx
    # del local_indices, chunk_top_indices

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
    query_positive_scores, global_indices, positive_to_queries, scores
):
    # ====================================================================
    # SIMPLIFIED QUERY-POSITIVE SCORE EXTRACTION
    # ====================================================================

    # Extract query-positive scores only if this chunk contains positives

    # Convert global indices to set for faster lookup
    global_indices_set = set(global_indices.cpu().numpy())

    # Collect all (query_idx, local_idx, qrel_idx) tuples for this chunk
    batch_queries = []
    batch_locals = []
    batch_qrels = []

    # Build global_to_local mapping for this chunk
    global_to_local = {g.item(): i for i, g in enumerate(global_indices)}

    # Check which positives are in this chunk
    for global_idx in global_indices_set:
        if global_idx in positive_to_queries:
            local_idx = global_to_local[global_idx]
            for query_idx, qrel_idx in positive_to_queries[global_idx]:
                batch_queries.append(query_idx)
                batch_locals.append(local_idx)
                batch_qrels.append(qrel_idx)

    # Vectorized extraction: gather all scores at once
    if batch_queries:
        extracted_scores = scores[batch_queries, batch_locals]
        query_positive_scores[batch_qrels] = extracted_scores

    return query_positive_scores


def search2(
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
    print_every=10**5,
):
    """
    Search for top-k documents and compute query-positive scores.

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
        print_every: Print progress every N documents

    Returns:
        tuple: (top_scores, top_indices, query_positive_scores)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    N_queries = query_embeddings.shape[0]
    N_corpus = len(corpus_dataset)

    top_scores = torch.full(
        (N_queries, top_k), -float("inf"), device=query_embeddings.device
    )
    top_indices = torch.full(
        (N_queries, top_k), -1, dtype=torch.long, device=query_embeddings.device
    )

    # Vectorized preparation for query-positive score extraction
    # Build mapping from corpus_id to corpus_index (only for positives)
    corpus_id_to_idx = {pid: idx for idx, pid in enumerate(corpus_ids[:n_positives])}

    # Vectorized mapping: convert qrels_positive_ids to corpus indices
    qrels_positive_ids_array = np.array(qrels_positive_ids)
    qrels_query_ids_array = np.array(qrels_query_ids)

    # Map positive IDs to corpus indices (vectorized)
    corpus_indices = np.array(
        [corpus_id_to_idx.get(pid, -1) for pid in qrels_positive_ids]
    )
    # Filter out entries where positive is not found or is beyond n_positives
    valid_mask = corpus_indices >= 0

    # Store the mapping information for later use
    qrels_corpus_indices = corpus_indices[valid_mask]
    qrels_query_indices = np.array(
        [unique_query_id_to_idx[qid] for qid in qrels_query_ids_array[valid_mask]]
    )
    qrels_indices_filtered = np.where(valid_mask)[0]

    # Initialize query-positive scores array
    query_positive_scores = torch.zeros(
        len(qrels_query_ids), device=query_embeddings.device
    )

    interval = print_every // chunk_size + 1
    dist.barrier()
    start = time.time()
    if rank == 0:
        print(f"Using chunk_size: {chunk_size}")
        print(
            f"Will extract query-positive scores for {len(qrels_query_ids)} qrels pairs"
        )

    for i, chunk_idx in enumerate(range(0, N_corpus, chunk_size)):

        dist.barrier()
        torch.cuda.empty_cache()

        if (i + 1) % interval == 0 and rank == 0:
            print(
                f"processed {chunk_idx//10**3}k/{N_corpus//10**3}k samples in {(time.time()-start)/60} mins"
            )

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
            collate_fn=collate_fn,
        )

        local_corpus_chunk, local_indices = encode(
            model,
            corpus_loader,
            prompt_type=PromptType.document,
            world_size=world_size,
            divided_by_chunks=True,
        )

        # Compute global indices for this chunk
        global_indices = local_indices + chunk_idx

        # Compute similarity scores for all query-document pairs in this chunk
        scores = torch.matmul(query_embeddings, local_corpus_chunk.T)

        # Extract query-positive scores only if this chunk contains positives
        if chunk_idx < n_positives:
            # Convert to numpy for efficient indexing
            global_indices_np = global_indices.cpu().numpy()

            # Find which qrels entries have their positive in this chunk
            # Vectorized: check if qrels_corpus_indices are in current chunk's global_indices
            in_chunk_mask = np.isin(qrels_corpus_indices, global_indices_np)

            if in_chunk_mask.any():
                # Get the qrels entries that need processing in this chunk
                chunk_qrels_indices = qrels_indices_filtered[in_chunk_mask]
                chunk_corpus_indices = qrels_corpus_indices[in_chunk_mask]
                chunk_query_indices = qrels_query_indices[in_chunk_mask]

                # Find local indices within the chunk for these corpus indices
                # Build a mapping from global_idx to local_idx for this chunk
                global_to_local = {
                    global_idx.item(): local_idx
                    for local_idx, global_idx in enumerate(global_indices)
                }
                local_indices_in_chunk = np.array(
                    [global_to_local[corpus_idx] for corpus_idx in chunk_corpus_indices]
                )

                # Vectorized extraction: gather scores using advanced indexing
                # scores[chunk_query_indices, local_indices_in_chunk] gives us all the scores we need
                extracted_scores = scores[chunk_query_indices, local_indices_in_chunk]
                query_positive_scores[chunk_qrels_indices] = extracted_scores

        chunk_top_scores, chunk_top_indices = torch.topk(
            scores,
            k=min(top_k, scores.shape[1]),
            dim=1,
            largest=True,
        )

        chunk_absolute_indices = local_indices[chunk_top_indices] + chunk_idx
        combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
        combined_indices = torch.cat([top_indices, chunk_absolute_indices], dim=1)

        # Find the true global top-k among the combined results
        top_k_in_combined_scores, top_k_in_combined_indices = torch.topk(
            combined_scores,
            k=top_k,
            dim=1,
            largest=True,
        )

        top_scores = top_k_in_combined_scores
        top_indices = torch.gather(combined_indices, 1, top_k_in_combined_indices)

    dist.barrier()

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

    # Gather query-positive scores from all GPUs
    if rank == 0:
        print(f"\nGathering query-positive scores from all GPUs...")

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


@torch.inference_mode()
def encode(model, loader, world_size, prompt_type, divided_by_chunks=False):

    # distributed sampler will duplicate examples at the end
    indices = None
    if hasattr(loader.sampler, "indices"):
        indices = loader.sampler.indices
        assert isinstance(indices, list)

    num_samples = len(loader.dataset)
    embeddings = []

    for i, batch in enumerate(loader):

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
        embeddings.append(batch_embeddings.float())

    embeddings = torch.cat(embeddings, dim=0)
    indices = torch.tensor(indices, device=embeddings.device)

    if prompt_type == PromptType.document and divided_by_chunks:
        # if we are processing documents divided in chunks, we postpone the allgather
        # (in the distributed setup) and return the (local) indices
        return embeddings, indices

    if world_size > 1:
        gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
        dist.all_gather(gathered, embeddings)
        # Concatenate across ranks for this batch
        embeddings = torch.cat(gathered, dim=0)
        embeddings = embeddings[:num_samples]

        # Also gather indices if we tracked them
        if indices is not None:
            gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
            dist.all_gather(gathered_indices, indices)
            indices = torch.cat(gathered_indices, dim=0)
            indices = indices[:num_samples]

    # Restore original order
    sorted_positions = torch.argsort(indices)
    embeddings = embeddings[sorted_positions]
    return embeddings
