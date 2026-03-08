import contextlib
import torch
import os

# Tokenizer parallelism: safe to enable because the DataLoader uses
# multiprocessing_context="spawn" (no fork → no thread-pool deadlock).
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # OLD: disabled for fork safety
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import time
import json
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.arguments import parse_args
from utils.helpers import (
    print_memory_consumed,
    save_model,
    get_cpt_steps,
    get_train_ds_config,
)
from models.t5gemma2model import get_model_t5gemma2_model
from utils.optimizer import get_scheduler_optimizer
from utils.create_datasets import (
    create_hard_negatives_datasets,
    create_and_tokenize_hard_negatives_datasets,
    create_hard_negatives_datasets_from_pretokenized,
    create_per_dataset_from_pretokenized,
    DATASET_SUBSET,
    get_eval_tasks,
)
from utils.dataloader_helpers import (
    collate_fn_with_hard_negatives,
    collate_fn_pretokenized_fast_pad_v2,
    DatasetAwareSampler,
    MultiDatasetLoader,
)
from utils.losses import (
    EmbeddingGemmaLossDistributed,
    EmbeddingGemmaLossHardNegatives,
    F2LLMLoss,
)
from typing import Callable
import torch.nn.functional as TF
from functools import partial

from huggingface_hub import login as hf_login

try:
    import deepspeed
    from transformers.integrations import HfDeepSpeedConfig

    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False

from inference.test_retrieval_ddp_update import evaluate_retrieval
from utils.create_datasets import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)
from models.modules import last_token_pool, add_pooling_layers, mean_pool
from datetime import timedelta
from tasks import RETRIEVAL_SUBSET, NAME_TO_TASK_TYPE
from tasks.task_categories import BINARY_CLASSIFICATION_TASKS


class CudaDataPrefetcher:
    """Prefetches batches onto GPU using a side CUDA stream.

    While the current batch is being processed on the default stream, the next
    batch is asynchronously transferred to GPU on a separate stream.  This
    hides the host-to-device copy latency almost entirely.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter = None
        self.next_batch = None

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                key: (
                    val.to(self.device, non_blocking=True)
                    if isinstance(val, torch.Tensor)
                    else val
                )
                for key, val in batch.items()
            }

    def next(self):
        if self.iter is None:
            self.iter = iter(self.loader)
            self._preload()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            # Ensure tensors record the dependency on the side stream
            for val in batch.values():
                if isinstance(val, torch.Tensor):
                    val.record_stream(torch.cuda.current_stream(self.device))
            self._preload()
        return batch


class Trainer:
    def __init__(
        self,
        len_dataloader,
        model,
        args,
        task_type=None,
        lora_modules=None,
    ):

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(self.local_rank)

        self.model = model
        assert self.rank == RANK
        assert self.world_size == WORLD_SIZE

        if self.rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.activation_checkpointing:
            # Resolve the underlying transformer model for checkpointing.
            # EmbeddingT5Gemma2* variants expose it as .encoder;
            # EncoderWithPooling (Qwen3, embeddinggemma) wraps it as .model.
            if hasattr(self.model, "encoder"):
                _base = self.model.encoder
            elif hasattr(self.model, "model"):
                _base = self.model.model
            else:
                _base = self.model

            # Disable cache first
            if hasattr(_base, "config"):
                _base.config.use_cache = False

            # Enable PyTorch gradient checkpointing on all layers first.
            _base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            # Selective checkpointing: keep only one in every N layers
            # checkpointed (see train_deepspeed.py for full rationale).
            interval = getattr(args, "checkpoint_layers_interval", 1)
            if interval > 1 and hasattr(_base, "layers"):
                for i, layer in enumerate(_base.layers):
                    if i % interval != 0:
                        layer.gradient_checkpointing = False
                n_ckpt = sum(
                    getattr(l, "gradient_checkpointing", False) for l in _base.layers
                )
                print(
                    f"Selective activation checkpointing: {n_ckpt}/{len(_base.layers)} "
                    f"layers checkpointed (interval={interval})"
                )

        if args.use_lora:
            peft_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        print_memory_consumed(message="memory consumed before loading model")

        self.use_deepspeed = getattr(args, "deepspeed", False)

        if self.use_deepspeed:
            if not _DEEPSPEED_AVAILABLE:
                raise ImportError("--deepspeed requires the 'deepspeed' package.")

            ds_config = get_train_ds_config(
                train_batch_size=args.batch_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                stage=getattr(args, "deepspeed_stage", 2),
                max_norm=args.clip_grad_thresh,
            )

            # Required for ZeRO-3 to partition the model during from_pretrained.
            # Must be created before deepspeed.initialize().
            if ds_config["zero_optimization"]["stage"] == 3:
                _dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 (kept alive)

            self.optimizer, self.lr_scheduler = get_scheduler_optimizer(
                self.model,
                args,
                len_dataloader,
            )

            print_memory_consumed(message="before deepspeed.initialize")
            self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config=ds_config,
                lr_scheduler=self.lr_scheduler,
                dist_init_required=False,
            )
            self.model.train()
            print_memory_consumed(message="after deepspeed.initialize")
        else:
            # 3. Move your model to the device
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                static_graph=True,  # cache allreduce schedule (graph never changes)
                gradient_as_bucket_view=True,  # avoid gradient-to-bucket copy
            )

            if not args.no_compile:
                if RANK == 0:
                    print(
                        "wrapping with torch.compile (actual compilation deferred to first forward pass)..."
                    )
                self.model = torch.compile(self.model)
            print_memory_consumed(message="memory consumed after loading model")

            self.optimizer, self.lr_scheduler = get_scheduler_optimizer(
                self.model,
                args,
                len_dataloader,
                fused=True,
            )

        # DeepSpeed handles mixed precision internally; DDP uses torch.autocast.
        self.autocast_ctx = (
            contextlib.nullcontext()
            if self.use_deepspeed
            else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        )

        # Bind the correct backward implementation once at init so the hot loop
        # dispatches with a direct attribute lookup — no branch per step.
        self._backward_step = (
            self._backward_step_ds if self.use_deepspeed else self._backward_step_ddp
        )

    def _backward_step_ds(self, loss: torch.Tensor, args) -> None:
        """DeepSpeed backward + update (clip/zero_grad/lr_step handled internally)."""
        self.model.backward(loss)
        self.model.step()

    def _backward_step_ddp(self, loss: torch.Tensor, args) -> None:
        """DDP backward + update."""
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_thresh)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train(
        self,
        args: ArgumentParser,
        train_loader: DataLoader,
        loss_fn: Callable,
        evaluator,
    ):

        filename = ""
        if args.out_filename != "":
            filename = "_" + args.out_filename

        eval_steps, _ = get_cpt_steps(
            int(args.eval_steps), args.max_train_steps, logspace=False
        )
        checkpointing_steps, _ = get_cpt_steps(
            args.checkpointing_steps, args.max_train_steps, logspace=False
        )
        log_steps, _ = get_cpt_steps(
            int(args.logging_steps), args.max_train_steps, logspace=False
        )

        stats = defaultdict(dict)
        stats["train_params"] = {
            "num_epochs": args.num_train_epochs,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
        }
        if args.use_lora:
            stats["train_params"]["lora"] = {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            }
        else:
            stats["train_params"]["lora"] = False

        if args.measure_baselines:
            self.model.eval()
            results, summary = evaluator.evaluate(
                self.model, batch_size=args.per_device_eval_batch_size
            )
            self.model.train()
            stats["test_perf"][0] = summary
            if RANK == 0:
                print(summary)

        if RANK == 0:
            print("log_steps:", log_steps)
            print("eval_steps", eval_steps)
            print("***** Running training *****")
            print(f"  Num Epochs = {args.num_train_epochs}")
            print(f"  Learning rate = {args.learning_rate}")
            print(f"  Weight Decay = {args.weight_decay}")
            if args.use_lora:
                print(f"  Lora Rank = {args.lora_rank}")
                print(f"  Lora Alpha = {args.lora_alpha}")
                print(f"  Lora Dropout = {args.lora_dropout}")
            print(f"  Batch size per device = {args.per_device_train_batch_size}")
            print(
                f"  Total batch size (w. parallel, distributed & accumulation) = {args.batch_size}"
            )
            print(f"  world size = {WORLD_SIZE}")
            print(f"  len_dataloader = {len(train_loader)}")
            print(f"  Total optimization steps = {args.max_train_steps}")
            print(f"  Log steps number = {len(log_steps)}")

            print("memory before train run")
            print_memory_consumed(rank=RANK)
            print("\nstart training...")

        completed_steps = 0
        total_loss = 0
        total_time = 0
        previous_cpt = 0  # used by deepspeed path for per-interval loss averaging

        is_f2llm = isinstance(loss_fn, F2LLMLoss)
        # Pre-compute set of retrieval task names for per-batch inbatch decision
        _retrieval_types = {
            "Retrieval",
            "PairClassification",
            "STS",
            "Reranking",
            "Summarization",
        }
        _inbatch_tasks = (
            frozenset(
                name
                for name, ttype in NAME_TO_TASK_TYPE.items()
                if ttype in _retrieval_types
            )
            if is_f2llm
            else frozenset()
        )

        start = time.time()
        for epoch in range(args.num_train_epochs):

            self.model.train()
            # gradient accumulation step may not finish with a proper update at the end of the epoch so we call zero grad here
            # OLD: self.optimizer.zero_grad()
            self.optimizer.zero_grad(set_to_none=True)

            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # --- Non-blocking H2D transfer (overlaps with previous step's compute) ---
            for index, batch in enumerate(train_loader):
                if index == 0 and epoch == 0 and RANK == 0:
                    print(
                        "first batch received from dataloader, entering forward pass..."
                    )
                batch = {
                    key: (
                        val.to(self.device, non_blocking=True)
                        if isinstance(val, torch.Tensor)
                        else val
                    )
                    for key, val in batch.items()
                }
                # --- END H2D ---

                # prefetcher = CudaDataPrefetcher(train_loader, self.device)
                # batch = prefetcher.next()
                # index = 0
                # while batch is not None:

                query_inputs = batch["query_token_ids"]
                query_mask = batch["query_attention_mask"]
                all_doc_inputs = batch["all_doc_token_ids"]
                all_doc_mask = batch["all_doc_attention_mask"]
                doc_ids = batch["pos_ids"]
                query_ids = batch["query_ids"]
                num_neg = batch["num_hard_negatives"]

                with self.autocast_ctx:
                    B = query_inputs.shape[0]
                    query_embeddings = self.model(
                        input_ids=query_inputs, attention_mask=query_mask
                    )
                    all_doc_embeddings = self.model(
                        input_ids=all_doc_inputs, attention_mask=all_doc_mask
                    )
                    doc_embeddings = all_doc_embeddings[:B]
                    neg_embeddings = all_doc_embeddings[B:].view(B, num_neg, -1)

                    if is_f2llm:
                        ds_name = batch.get("dataset_name", "")
                        _ib = ds_name in _inbatch_tasks
                        loss = loss_fn(
                            query_embeddings=query_embeddings,
                            doc_embeddings=doc_embeddings,
                            hard_neg_embeddings=neg_embeddings,
                            doc_ids=doc_ids,
                            query_ids=query_ids,
                            use_inbatch=_ib,
                        )
                    else:
                        loss = loss_fn(
                            query_embeddings=query_embeddings,
                            doc_embeddings=doc_embeddings,
                            hard_neg_embeddings=neg_embeddings,
                            doc_ids=doc_ids,
                            query_ids=query_ids,
                        )

                self._backward_step(loss, args)

                total_loss += loss.detach().float()
                completed_steps += 1

                if completed_steps == 1 and RANK == 0:
                    print(f"first step completed in {time.time()-start:.1f}s")

                if completed_steps in log_steps or completed_steps == 10:

                    if WORLD_SIZE > 1:
                        total_loss = total_loss.reshape(1)
                        dist.all_reduce(total_loss)

                    avg_loss = (
                        total_loss.item()
                        / WORLD_SIZE
                        / (completed_steps - previous_cpt)
                    )
                    previous_cpt = completed_steps
                    total_loss = 0

                    if RANK == 0:
                        stats["loss"][completed_steps] = avg_loss
                        print(f"log step: {completed_steps}/{log_steps[-1]}")
                        print_memory_consumed(rank=RANK)

                        total_time = time.time() - start

                        print(
                            f"LR: {self.lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, \
                                Time: {int(total_time//3600)} h {(total_time%3600)/60: .2f} min"
                        )

                        with open(
                            f"{args.output_dir}/train_logs{filename}.json", "w"
                        ) as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in eval_steps:
                    self.model.eval()
                    _, summary = evaluator.evaluate(
                        self.model, batch_size=args.per_device_eval_batch_size
                    )
                    self.model.train()

                    if RANK == 0:
                        print(f"iter {completed_steps}.")
                        stats["test_perf"][completed_steps] = summary
                        with open(
                            f"{args.output_dir}/train_logs{filename}.json", "w"
                        ) as f:
                            json.dump(stats, f, indent=4)

                if completed_steps in checkpointing_steps and args.save_checkpoint:
                    if RANK == 0:
                        print("saving checkpoint")

                    output_dir = f"ckpts{filename}/step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    save_model(
                        self.model, output_dir, RANK=RANK, dist_type=args.dist_type
                    )

            total_time = time.time() - start
            stats["total_time_min"] = total_time / 60
            if RANK == 0:
                with open(f"{args.output_dir}/train_logs{filename}.json", "w") as f:
                    json.dump(stats, f, indent=4)

            eval_tasks = get_eval_tasks(
                "mteb_eng_v2",
                task_types=["Retrieval", "Summarization", "STS", "Reranking"],
            )
            evaluator.update_datasets(eval_tasks)
            results, summary = evaluator.evaluate(
                self.model, batch_size=args.per_device_eval_batch_size
            )
            stats["mteb_eng_v2_full"] = summary
            with open(f"{args.output_dir}/train_logs{filename}.json", "w") as f:
                json.dump(stats, f, indent=4)

            output_dir = f"epoch_{epoch+1}{filename}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_model(self.model, output_dir, RANK=RANK, dist_type=args.dist_type)


def main():
    args = parse_args()

    # Login to Hugging Face for gated models (read token from .hf_token, gitignored)
    _hf_token_path = os.path.join(os.path.dirname(__file__), ".hf_token")
    if os.path.isfile(_hf_token_path):
        with open(_hf_token_path, "r") as f:
            token = f.read().strip()
        if token:
            hf_login(token=token)

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(minutes=30),
    )
    rank = dist.get_rank()
    torch.cuda.set_device(LOCAL_RANK)
    torch.set_float32_matmul_precision("high")

    if args.use_deepspeed:
        # need activation checkpointing otherwise oom
        args.activation_checkpointing = True
        args.per_device_train_batch_size = min(32, args.per_device_train_batch_size)
        if RANK == 0:
            print("USE DEEPSPEED: set activation checkpointing to TRUE")
            print("USE DEEPSPEED: set min batch_size to 32")

    args.batch_size = WORLD_SIZE * args.per_device_train_batch_size
    args.gradient_accumulation_steps = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if RANK == 0:
        print("loading train set ")
        start = time.time()

    if "t5gemma-2" in args.model_name_or_path.lower():
        instruction_template = instruction_template_embeddinggemma
        add_special_tokens = True
        eot_id = None

        model, task_type, lora_modules = get_model_t5gemma2_model(
            model_name_or_path=args.model_name_or_path,
            activation_checkpointing=args.activation_checkpointing,
            attention_pooling=args.attention_pooling,
            cls_query_pooling=args.cls_query_pooling,
            attention_dim=args.attention_dim,
            attn_implementation=args.attn_implementation,
        )

        model_name = "t5gemma2"
        if args.attention_pooling:
            model_name += "_attn_pooling"
        if args.cls_query_pooling:
            model_name += "_cls_query"

    elif "qwen3" in args.model_name_or_path.lower():
        model_name = "qwen3"
        instruction_template = instruction_template_qwen3
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
        ).to("cuda")
        model = add_pooling_layers(model, pool_fn=last_token_pool)

        task_type = None
        lora_modules = [
            "q_proj",
            "o_proj",
            "v_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        if RANK == 0:
            print(f"qwen3 model loaded (eval_only={args.eval_only})")
    elif "embeddinggemma" in args.model_name_or_path.lower():

        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
        ).to("cuda")
        model = add_pooling_layers(model, pool_fn=mean_pool)

        args.use_lora = False
        args.eval_only = True
        task_type = None
        lora_modules = None

    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3' or 'embeddinggemma'."
        )

    if RANK == 0:
        print(f"length_strategy={args.length_strategy}, max_seq_len={args.max_seq_len}")

    train_list = None  # defaults to all
    if args.train_subset == "reduced":
        train_list = DATASET_SUBSET
    elif args.train_subset == "retrieval":
        train_list = RETRIEVAL_SUBSET

    teacher_model = args.negatives_dir.split("/")[-1]

    # if args.tokenize_dataset:
    #     train_dataset = create_and_tokenize_hard_negatives_datasets(
    #         base_dir=args.negatives_dir,
    #         num_hard_negatives=args.num_hard_negatives,
    #         tokenizer=tokenizer,
    #         instruction_template=instruction_template,
    #         add_special_tokens=add_special_tokens,
    #         rank=RANK,
    #         datasets_subset=train_list,
    #         max_seq_len=args.max_seq_len if args.length_strategy == "filter" else None,
    #     )
    # else:
    #     train_dataset = create_hard_negatives_datasets(
    #         base_dir=args.negatives_dir,
    #         num_hard_negatives=args.num_hard_negatives,
    #         tokenizer=tokenizer,
    #         instruction_template=instruction_template,
    #         rank=RANK,
    #         datasets_subset=train_list,
    #         max_seq_len=args.max_seq_len if args.length_strategy == "filter" else None,
    #     )

    train_dataset = create_hard_negatives_datasets_from_pretokenized(
        base_dir=args.negatives_dir,
        rank=RANK,
        datasets_subset=train_list,
    )

    dist.barrier()
    if RANK == 0:
        print(f"datasets prepared in {time.time()-start:.1f}s")
        print("dataloader preparation")

    # ------------------------------------------------------------------
    # F2LLM multi-dataset path: per-dataset DataLoaders + MultiDatasetLoader
    # ------------------------------------------------------------------
    if args.batch_strategy == "f2llm_multi":
        per_dataset_dict = create_per_dataset_from_pretokenized(
            base_dir=args.negatives_dir,
            rank=RANK,
            datasets_subset=train_list,
        )

        binary_set = set(BINARY_CLASSIFICATION_TASKS)
        args.num_workers = int(args.num_workers * 1.25)

        loaders: dict[str, DataLoader] = {}
        for ds_name, ds in per_dataset_dict.items():
            # Binary classification tasks: 1 hard negative
            # Everything else: full num_hard_negatives
            k = 1 if ds_name in binary_set else args.num_hard_negatives

            ds_sampler = DistributedSampler(
                ds,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                seed=42,
            )
            ds_collate = partial(
                collate_fn_pretokenized_fast_pad_v2,
                pad_token_id=tokenizer.pad_token_id,
                num_hard_negatives=k,
                padding_side="right",
                eot_id=eot_id,
            )
            loaders[ds_name] = DataLoader(
                ds,
                batch_size=args.per_device_train_batch_size,
                sampler=ds_sampler,
                collate_fn=ds_collate,
                num_workers=min(args.num_workers, 2),
                pin_memory=True,
                persistent_workers=min(args.num_workers, 2) > 0,
                prefetch_factor=4 if min(args.num_workers, 2) > 0 else None,
                multiprocessing_context=(
                    "spawn" if min(args.num_workers, 2) > 0 else None
                ),
            )

        train_loader = MultiDatasetLoader(loaders)

    # ------------------------------------------------------------------
    # Standard path: single concatenated dataset
    # ------------------------------------------------------------------
    else:
        if args.batch_strategy in ("sequential", "grouped"):
            sampler = DatasetAwareSampler(
                train_dataset,
                batch_size=args.per_device_train_batch_size,
                strategy=args.batch_strategy,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                seed=42,
            )
        else:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=False,
                seed=42,
            )

        collate_fn = partial(
            collate_fn_pretokenized_fast_pad_v2,
            pad_token_id=tokenizer.pad_token_id,
            num_hard_negatives=args.num_hard_negatives,
            padding_side="right",
            eot_id=eot_id,
        )

        # num workers can be slightly more that tot/num_ranks (empirical factor)
        args.num_workers = int(args.num_workers * 1.25)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
        )

    eval_tasks = get_eval_tasks(args.eval_set)
    evaluator = evaluate_retrieval(
        tasks=eval_tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        add_special_tokens=add_special_tokens,
        eot_id=eot_id,
        max_samples=1_000_000,
    )

    if args.loss_type == "f2llm":
        loss_fn = F2LLMLoss(temperature=args.f2llm_temperature)
    else:
        loss_fn = EmbeddingGemmaLossHardNegatives(
            temperature=0.07, num_hard_negatives=args.num_hard_negatives
        )
        if WORLD_SIZE > 1 and args.distributed_loss:
            loss_fn = EmbeddingGemmaLossDistributed(temperature=0.07)

    dist.barrier()

    _ds_prefix = "deepspeed_" if getattr(args, "deepspeed", False) else ""
    suffix = f"{_ds_prefix}{model_name}_train-{teacher_model}_gpus{WORLD_SIZE}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_{args.batch_strategy}"
    if args.out_filename:
        args.out_filename = f"{args.out_filename}_{suffix}"
    else:
        args.out_filename = suffix

    trainer = Trainer(
        len_dataloader=len(train_loader),
        model=model,
        task_type=task_type,
        lora_modules=lora_modules,
        args=args,
    )

    if args.eval_only:
        results, summary = evaluator.evaluate(
            trainer.model, batch_size=args.per_device_eval_batch_size
        )

        if RANK == 0:
            print(results)

        label = args.eval_set
        with open(f"{model_name}_{label}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
    else:
        dist.barrier()
        trainer.train(
            args=args,
            train_loader=train_loader,
            loss_fn=loss_fn,
            evaluator=evaluator,
        )
        dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
