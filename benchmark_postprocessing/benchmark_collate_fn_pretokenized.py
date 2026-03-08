import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import time
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import AutoTokenizer
from functools import partial
from datetime import timedelta

from utils.arguments import parse_args
from utils.create_datasets import (
    create_hard_negatives_datasets_from_pretokenized,
    DATASET_SUBSET,
)
from utils.dataloader_helpers import (
    collate_fn_pretokenized,
    collate_fn_pretokenized_fast_pad,
    collate_fn_pretokenized_fast_pad_v2,
    DatasetAwareSampler,
)

from huggingface_hub import login as hf_login


def main():

    args = parse_args()
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(LOCAL_RANK)

    _hf_token_path = os.path.join(os.path.dirname(__file__), ".hf_token")
    if os.path.isfile(_hf_token_path):
        with open(_hf_token_path, "r") as f:
            token = f.read().strip()
        if token:
            hf_login(token=token)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", LOCAL_RANK)

    args.batch_size = WORLD_SIZE * args.per_device_train_batch_size
    args.gradient_accumulation_steps = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_list = DATASET_SUBSET

    # Pre-tokenized dataset: tokens are already computed, collate only pads.
    train_dataset = create_hard_negatives_datasets_from_pretokenized(
        base_dir=args.negatives_dir,
        rank=RANK,
        datasets_subset=train_list,
    )

    sampler = DatasetAwareSampler(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        strategy="grouped",
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        seed=42,
    )

    # ------------------------------------------------------------------ #
    #  Collate variants to benchmark                                       #
    # ------------------------------------------------------------------ #
    NUM_BENCH_BATCHES = 500
    NUM_WARMUP_BATCHES = 20

    collate_variants = {
        "collate_fn_pretokenized": collate_fn_pretokenized,
        "collate_fn_pretokenized_fast_pad": collate_fn_pretokenized_fast_pad,
        "collate_fn_pretokenized_fast_pad_v2": collate_fn_pretokenized_fast_pad_v2,
    }

    # Shared kwargs for both collate functions
    collate_kwargs = dict(
        pad_token_id=tokenizer.pad_token_id,
        num_hard_negatives=args.num_hard_negatives,
        padding_side="right",
        eot_id=None,
    )

    # ------------------------------------------------------------------ #
    #  Single-process benchmark (num_workers=0)                           #
    #  Both variants see the exact same batches; useful for apples-to-    #
    #  apples CPU-only collate timing without I/O overlap.                #
    # ------------------------------------------------------------------ #
    # if args.num_workers == 0:
    #     if RANK == 0:
    #         print(f"\n{'='*80}")
    #         print(f"  Single-process benchmark  (num_workers=0)")
    #         print(f"{'='*80}")

    #     single_results: dict[str, tuple[float, int]] = {}

    #     for variant_name, collate_func in collate_variants.items():
    #         if RANK == 0:
    #             print(f"\nbenchmarking variant: {variant_name}")

    #         collate_fn = partial(collate_func, **collate_kwargs)

    #         train_loader = DataLoader(
    #             train_dataset,
    #             batch_size=args.per_device_train_batch_size,
    #             sampler=sampler,
    #             collate_fn=collate_fn,
    #             num_workers=0,
    #             pin_memory=True,
    #             persistent_workers=False,
    #             prefetch_factor=None,
    #             multiprocessing_context=None,
    #         )

    #         # warmup
    #         sampler.set_epoch(0)
    #         for _i, _ in enumerate(train_loader):
    #             if _i >= NUM_WARMUP_BATCHES:
    #                 break

    #         sampler.set_epoch(0)
    #         start = time.perf_counter()
    #         n_batches = 0
    #         for index, batch in enumerate(train_loader):
    #             batch = {
    #                 key: val.to(device) if isinstance(val, torch.Tensor) else val
    #                 for key, val in batch.items()
    #             }
    #             n_batches += 1
    #             if index + 1 >= NUM_BENCH_BATCHES:
    #                 break
    #         single_results[variant_name] = (time.perf_counter() - start, n_batches)

    #     if RANK == 0:
    #         col_w = 14
    #         print(
    #             f"\n  {'variant':<38}  {'wall_s':>{col_w}}  {'ms/batch':>{col_w}}  {'batches':>8}  {'speedup':>8}"
    #         )
    #         print(f"  {'-'*38}  {'-'*col_w}  {'-'*col_w}  {'-'*8}  {'-'*8}")
    #         ref_dur, _ = single_results[list(single_results.keys())[0]]
    #         for name, (dur, nb) in single_results.items():
    #             ms_per = dur / nb * 1000
    #             speedup = ref_dur / dur
    #             print(
    #                 f"  {name:<38}  {dur:>{col_w}.3f}  {ms_per:>{col_w}.2f}  {nb:>8d}  {speedup:>7.2f}x"
    #             )

    # # ------------------------------------------------------------------ #
    # #  Wall-time benchmark (num_workers > 0)  –  production setting       #
    # #  Collate runs inside worker subprocesses; measures real throughput   #
    # #  including I/O overlap and prefetch.                                 #
    # # ------------------------------------------------------------------ #
    # else:
    if RANK == 0:
        print(f"\n{'='*80}")
        print(
            f"  Wall-time benchmark  (num_workers={args.num_workers},  production setting)"
        )
        print(f"{'='*80}")

    # ------------------------------------------------------------------ #
    #  Shared global warmup                                                #
    #  Runs BEFORE any variant so that Arrow/mmap page faults are         #
    #  paid equally by all variants.  Without this, the first variant     #
    #  pays cold-start I/O costs that make subsequent variants look        #
    #  artificially faster.                                                #
    # ------------------------------------------------------------------ #
    if RANK == 0:
        print(f"\nGlobal warmup ({NUM_WARMUP_BATCHES} batches, untimed)...")
    _warmup_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn_pretokenized, **collate_kwargs),
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    sampler.set_epoch(0)
    for _i, _ in enumerate(_warmup_loader):
        if _i >= NUM_WARMUP_BATCHES:
            break
    _warmup_loader._iterator = None
    del _warmup_loader
    if RANK == 0:
        print("Global warmup done.\n")

    wall_results: dict[str, tuple[float, int]] = {}

    for variant_name, collate_func in collate_variants.items():
        if RANK == 0:
            print(f"\nbenchmarking variant: {variant_name}")

        collate_fn = partial(collate_func, **collate_kwargs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="spawn",
        )

        # warmup: spin up worker pool
        sampler.set_epoch(0)
        for _i, _ in enumerate(train_loader):
            if _i >= NUM_WARMUP_BATCHES:
                break

        sampler.set_epoch(0)
        start = time.perf_counter()
        n_batches = 0
        for index, batch in enumerate(train_loader):
            batch = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            n_batches += 1
            if index + 1 >= NUM_BENCH_BATCHES:
                break
        wall_results[variant_name] = (time.perf_counter() - start, n_batches)

        # shut down workers before next variant
        train_loader._iterator = None
        del train_loader

    if RANK == 0:
        col_w = 14
        print(
            f"\n  {'variant':<38}  {'wall_s':>{col_w}}  {'ms/batch':>{col_w}}  {'batches':>8}  {'speedup':>8}"
        )
        print(f"  {'-'*38}  {'-'*col_w}  {'-'*col_w}  {'-'*8}  {'-'*8}")
        ref_dur, _ = wall_results[list(wall_results.keys())[0]]
        for name, (dur, nb) in wall_results.items():
            ms_per = dur / nb * 1000
            speedup = ref_dur / dur
            print(
                f"  {name:<38}  {dur:>{col_w}.3f}  {ms_per:>{col_w}.2f}  {nb:>8d}  {speedup:>7.2f}x"
            )


if __name__ == "__main__":
    main()
