import torch
import torch.distributed as dist
import numpy as np
import os


def return_formatted(ndata):
    for threshold, suffix in [(10**6, "M"), (10**3, "k")]:
        if ndata >= threshold:
            return f"{ndata / threshold:.2f} {suffix}"
    return str(ndata)


def print_memory_consumed(message="", rank=0):
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if rank == 0:
        print(f"CUDA mem allocated {message}: {allocated:.2f} GB")
        print(f"CUDA mem reserved {message}: {reserved:.2f} GB")


def _print_ram(label: str, rank: int = 0):
    """Print RAM usage on rank 0.

    Reports two complementary metrics:
      - VmRSS  (from /proc/self/status): RAM used by *this process* only
        (resident set size). Useful to see which step in the loader is
        responsible for memory growth.
      - available (from psutil): free RAM across the *entire system*,
        accounting for all processes and OS caches. Useful to know how
        close the machine is to OOM (especially with multiple DDP ranks).
    """
    if rank != 0:
        return
    # --- Per-process RSS (this process only) ---
    rss_gb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1024**2
                    break
    except FileNotFoundError:
        import resource

        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    # --- System-wide available RAM (all processes) ---
    # avail_gb = None
    # try:
    #     import psutil

    #     avail_gb = psutil.virtual_memory().available / 1024**3
    # except ImportError:
    #     pass

    parts = [f"[RAM] {label}:"]
    if rss_gb is not None:
        parts.append(f"process RSS {rss_gb:.2f} GB")
    # if avail_gb is not None:
    #     parts.append(f"system available {avail_gb:.2f} GB")
    print("  ".join(parts))


def get_cpt_steps(nsteps, max_train_steps, logspace=True):

    if logspace:
        steps = np.unique(
            np.around(np.geomspace(1, max_train_steps, nsteps, endpoint=False)).astype(
                int
            )
        )
        step = None
    else:
        step = max(1, int(np.around(max_train_steps / nsteps)))

        steps = np.arange(0, max_train_steps, step).astype(int)

    return steps, step


def save_model(model, output_dir, RANK, dist_type="ddp"):
    """Save model with proper unwrapping of DDP/FSDP/compile wrappers"""

    # 1. Handle FSDP Saving Separately (needs the wrapped model)
    if dist_type == "fsdp":
        # Assuming FSDP, FullStateDictConfig, StateDictType are imported/defined
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Get state dict from the wrapped FSDP model
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state_dict = model.state_dict()

        # Save only from rank 0
        if RANK == 0:
            os.makedirs(output_dir, exist_ok=True)
            # The wrapped FSDP model must be used for save_pretrained (HF/PEFT)
            # as it knows how to handle the sharded state_dict it just gathered.
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        return  # Exit the function after FSDP saving

    # 2. Handle DDP/DataParallel/Compile/Plain PyTorch Saving (needs the unwrapped model)

    # Unwrap all layers (DDP, DataParallel, torch.compile)
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    while isinstance(model, options):
        model = model.module

    # Unwrap torch.compile
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    # Get state dict from the fully unwrapped model
    state_dict = model.state_dict()

    # Save only from rank 0
    if RANK == 0:
        # Check if it's a PEFT model (has save_pretrained for LoRA)
        if hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
            # PEFT/LoRA model
            model.save_pretrained(output_dir, state_dict=state_dict)
        elif hasattr(model, "save_pretrained"):
            # HuggingFace model
            model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            # Plain PyTorch model
            os.makedirs(output_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    if dist.is_initialized():
        dist.barrier()
