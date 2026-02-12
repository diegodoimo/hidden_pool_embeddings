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
from utils.helpers import print_memory_consumed

# ***********************************************************************************************


def collate_fn_with_padding(batch, pad_token_id=0, padding_side="right", tokenizer = None, eot_id =None):

    input_text = [item["prompt"] for item in batch]
    tokens = tokenizer(
        input_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    query_token_ids = [torch.tensor(tok + [eot_id]) for tok in tokens]

    #query_token_ids = [torch.tensor(item["input_ids"]) for item in batch]
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


def search(
    model,
    query_embeddings,
    corpus_dataset,
    collate_fn,
    top_k=100,
    batch_size=64,
    chunk_size=10**4,
    print_every = 10**5,
    precomputed_doc_embeddings=None,
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    N_queries = query_embeddings.shape[0]
    N_corpus = len(corpus_dataset)

    # IMPORTANT: When using precomputed_doc_embeddings, they are fully replicated across all GPUs
    # (all_gather was performed during their creation). This means the similarity computation
    # will use world_size times more memory than distributed encoding.
    # Adjust chunk_size to compensate.
    if precomputed_doc_embeddings is not None:
        adjusted_chunk_size = max(1000, chunk_size // world_size)
        if rank == 0:
            print(f"Adjusting chunk_size for precomputed embeddings: {chunk_size} -> {adjusted_chunk_size} (divided by world_size={world_size})")
        chunk_size = adjusted_chunk_size

    top_scores = torch.full((N_queries, top_k), -float("inf"), device=query_embeddings.device)
    top_indices = torch.full(
        (N_queries, top_k), -1, dtype=torch.long, device=query_embeddings.device
    )
    interval = print_every // chunk_size + 1 
    dist.barrier()
    start = time.time()
    if rank ==0:
        print(f"Using chunk_size: {chunk_size}")
    for i, chunk_idx in enumerate(range(0, N_corpus, chunk_size)):
        
        dist.barrier()
        torch.cuda.empty_cache()

        if (i+1) % interval == 0 and rank == 0:
            print(f"processed {chunk_idx//10**3}k/{N_corpus//10**3}k samples in {(time.time()-start)/60} mins")

        if precomputed_doc_embeddings is not None:
            # Use pre-computed embeddings (skip encoding step)
            chunk_end = min(chunk_idx + chunk_size, N_corpus)
            local_corpus_chunk = precomputed_doc_embeddings[chunk_idx:chunk_end]
            local_indicies = torch.arange(chunk_end - chunk_idx, device=local_corpus_chunk.device)

            if rank ==0:
                print(f"iter {i}")
                print_memory_consumed(rank = rank)


        else:
            # Compute embeddings on-the-fly (original behavior)
            # IN THE DISTRIBUTERD SETUP WE ARE NOT HANDLING WELL THE LAST REPEATED SAMPLES. 
            subcorpus = corpus_dataset.select(range(chunk_idx, min(chunk_idx + chunk_size, N_corpus)))

            sampler_corpus = LenghtSortedSampler(subcorpus)
            corpus_loader = DataLoader(
                subcorpus,
                sampler=sampler_corpus,
                batch_size=batch_size,
                num_workers=16,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            local_corpus_chunk, local_indicies = encode(
                model,
                corpus_loader,
                prompt_type=PromptType.document,
                world_size=world_size,
                divided_by_chunks=True,
            )

        scores = torch.matmul(query_embeddings, local_corpus_chunk.T)
        chunk_top_scores, chunk_top_indices = torch.topk(
            scores,
            k=min(top_k, scores.shape[1]),  # Use min(top_k, chunk_size)
            dim=1,
            largest=True,
        ) 

        # print(chunk_top_indices, chunk_top_indices.shape, local_indicies)
        chunk_absolute_indices = local_indicies[chunk_top_indices] + chunk_idx
        combined_scores = torch.cat([top_scores, chunk_top_scores], dim=1)
        combined_indices = torch.cat([top_indices, chunk_absolute_indices], dim=1)

        # Find the true global top-k among the combined results
        # Note: We need k=top_k
        top_k_in_combined_scores, top_k_in_combined_indices = torch.topk(
            combined_scores,
            k=top_k,
            dim=1,
            largest=True,
        )

        top_scores = top_k_in_combined_scores
        top_indices = torch.gather(combined_indices, 1, top_k_in_combined_indices)

    dist.barrier()

    # --- 4. Distributed Merging (if using multiple GPUs) ---
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

    return top_scores, top_indices


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


