import torch
import torch.distributed as dist
import numpy as np
import os


def print_memory_consumed(message="", rank=0):
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    if rank == 0:
        print(f"CUDA mem allocated {message}: {allocated} GB")
        print(f"CUDA mem reserved {message}: {reserved} GB")


def get_cpt_steps(nsteps, max_train_steps, logspace=True):

    if logspace:
        steps = np.unique(
            np.around(np.geomspace(1, max_train_steps, nsteps, endpoint=False)).astype(int)
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
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
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
