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

        steps = np.arange(step, max_train_steps + 1, step).astype(int)

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

    # Iteratively unwrap all layers (torch.compile, DDP, DataParallel) in any order.
    # compile wraps DDP, so _orig_mod must be checked at every level.
    ddp_types = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    changed = True
    while changed:
        changed = False
        if hasattr(model, "_orig_mod"):  # torch.compile
            model = model._orig_mod
            changed = True
        if isinstance(model, ddp_types):  # DDP / DataParallel
            model = model.module
            changed = True

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


def get_train_ds_config(
    train_batch_size,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    stage=2,
    max_norm=1.0,
):
    """Return a DeepSpeed config aligned with CodeFuse-Embeddings F2LLM.

    F2LLM reference (accelerate_config.yaml):
        zero_stage: 2, gradient_clipping: 1.0, mixed_precision: bf16,
        offload_optimizer_device: none, offload_param_device: none,
        zero3_init_flag: false.
    """

    zero_opt = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        # Explicit no-offloading (matches F2LLM)
        "offload_optimizer": {"device": "none", "pin_memory": False},
        "offload_param": {"device": "none", "pin_memory": False},
    }

    # Stage-3-only knobs — only added when actually using stage 3
    if stage == 3:
        zero_opt.update(
            {
                "sub_group_size": 1e9,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )

    # bf16: {enabled: True} is sufficient because all models are loaded in
    # bfloat16 (torch_dtype=torch.bfloat16). DeepSpeed keeps parameters,
    # activations, and ZeRO communication in bf16 without needing torch_autocast.
    # This matches F2LLM's setup (accelerate mixed_precision: bf16).
    # NOTE: torch_autocast and bf16 cannot coexist in the same DeepSpeed config.
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": zero_opt,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": max_norm,
        "steps_per_print": 1e3,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "wall_clock_breakdown": False,
        "prescale_gradients": False,
    }

    return ds_config


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


# def get_train_ds_config(
#     offload,
#     train_batch_size,
#     per_device_train_batch_size=1,
#     adam_offload=False,
#     stage=0,
#     bf16=True,
#     max_norm=1.0,
#     grad_accum_dtype=None,
#     disable_trace_cache=True,
# ):
#     device = "cpu" if offload else "none"
#     zero_opt_dict = {
#         "stage": stage,
#         "offload_param": {"device": device},
#         "offload_optimizer": {
#             "device": "cpu" if adam_offload else "none",
#             "pin_memory": True,
#         },
#         "sub_group_size": "auto",
#         "stage3_max_live_parameters": "auto",
#         "stage3_max_reuse_distance": "auto",
#         "stage3_param_persistence_threshold": "auto",
#         "stage3_prefetch_bucket_size": "auto",
#         "reduce_bucket_size": "auto",
#     }
#     if disable_trace_cache:
#         zero_opt_dict["stage3_prefetch_bucket_size"] = 0
#         zero_opt_dict["stage3_max_live_parameters"] = 0
#         zero_opt_dict["stage3_max_reuse_distance"] = 0

#     config = {
#         "steps_per_print": 100,
#         "zero_optimization": zero_opt_dict,
#         "bf16": {
#             "enabled": bf16,
#         },
#         "gradient_clipping": max_norm,
#         "prescale_gradients": False,
#         "wall_clock_breakdown": False,
#         "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
#         "train_micro_batch_size_per_gpu": per_device_train_batch_size,
#         "train_batch_size": train_batch_size,
#     }

#     return config
