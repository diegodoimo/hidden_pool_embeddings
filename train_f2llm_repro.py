from f2llm_repro.f2llm_train import accelerate_train, CLASSIFICATION_DATASETS
from f2llm_repro.model import F2LLM, F2LLMT5Gemma2
from transformers import AutoTokenizer, set_seed, get_scheduler
import os, json, random
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

import argparse
from functools import partial

from inference.test_retrieval_ddp_update import evaluate_retrieval as EvaluateRetrieval
from utils.create_datasets import (
    get_eval_tasks,
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """

    def __init__(self, loader_dict, accelerator):
        self.loader_dict = loader_dict
        for k, v in self.loader_dict.items():
            self.loader_dict[k] = accelerator.prepare(v)

    def __len__(self):
        return sum(len(v) for v in self.loader_dict.values())

    def reset_epoch(self, epoch):
        self.rng = random.Random(epoch)
        self.iters = {k: iter(v) for k, v in self.loader_dict.items()}
        self.names = list(self.iters.keys())
        self.weights = [len(self.loader_dict[k]) for k in self.names]

    def __iter__(self):
        while self.names:  # until every DataLoader is empty
            name = self.rng.choices(self.names, weights=self.weights)[
                0
            ]  # pick a data-source at random
            try:
                batch = next(self.iters[name])
                yield batch
            except StopIteration:
                idx = self.names.index(name)
                self.names.pop(idx)  # this dataset has no batch left
                self.weights.pop(idx)


def _stack(input_ids, max_len, tokenizer):
    data = [ids[:max_len] for ids in input_ids]  # input_ids: list of lists
    data = [
        (
            ids
            if ids[-1] == tokenizer.eos_token_id
            else ids[:-1] + [tokenizer.eos_token_id]
        )
        for ids in data
    ]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))  # (total_tokens,)
    return tensor.split(lens)  # list of 1-d tensors


def collate_fn(batch_raw, args, _stack, tokenizer, classification_datasets):
    """
    length of input_ids: bs * (2 + num_hard_neg)
    0 - bs-1: query input ids
    bs - 2*bs-1: passage input ids
    2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
    2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
    """
    num_hard_neg = (
        1
        if batch_raw[0]["dataset_name"] in classification_datasets
        else args.num_hard_neg
    )
    # select args.num_hard_neg hard negatives from a total of 24
    hard_neg_indices = (
        [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
    )
    input_ids = _stack(
        [s["query_input_ids"] for s in batch_raw]
        + [s["passage_input_ids"] for s in batch_raw]
        + [s[f"negative_{i+1}_input_ids"] for s in batch_raw for i in hard_neg_indices],
        args.max_seq_length,
        tokenizer,
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    # pad input ids to [bs, max_len]
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "seq_lens": seqlens,
        "attention_mask": attention_masks,
        "bs": len(batch_raw),
        "dataset_name": batch_raw[0]["dataset_name"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_hard_neg", type=int, default=7)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--checkpointing_steps", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=10**9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--measure_baselines",
        action="store_true",
        help="Evaluate model at step 0 before any training",
    )
    parser.add_argument("--num_processes", type=int, default=0)
    parser.add_argument(
        "--eval_set",
        type=str,
        default="mteb_retrieval_subset",
        help="Name of the eval task set passed to get_eval_tasks() for mid-training MTEB evals.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) used during MTEB evaluation.",
    )
    parser.add_argument(
        "--instruction_template",
        type=str,
        default="qwen3",
        choices=["qwen3", "embeddinggemma"],
        help="Instruction-template style used when encoding text for MTEB evaluation.",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    return args


def main():

    args = parse_args()

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        deepspeed_plugin=deepspeed_plugin,
    )

    args.num_processes = accelerator.num_processes
    accelerator.print(args)

    # Detect model family once; drives model class + evaluator settings.
    is_t5gemma2 = "t5gemma-2" in args.model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    set_seed(0)
    if accelerator.is_main_process:
        os.makedirs(f"{args.output_dir}", exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    train_datasets, valid_datasets = [], []
    accelerator.print("loading datasets")
    with accelerator.main_process_first():
        for f in sorted(
            f for f in os.listdir(args.train_data_path) if f.endswith(".parquet")
        ):
            dataset_name = f.split(".parquet")[0]
            accelerator.print(f"loading {dataset_name}")

            dataset = load_dataset(
                "parquet",
                data_files=os.path.join(args.train_data_path, f),
                cache_dir=args.cache_dir,
            )["train"]
            dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
            dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
            train_datasets.append((dataset_name, dataset["train"]))
            valid_datasets.append((dataset_name, dataset["test"]))

    collate_fn_partial = partial(
        collate_fn,
        args=args,
        _stack=_stack,
        tokenizer=tokenizer,
        classification_datasets=CLASSIFICATION_DATASETS,
    )

    train_loaders = {
        name: DataLoader(
            ds,
            shuffle=True,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn_partial,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for name, ds in train_datasets
    }
    valid_loaders = {
        name: DataLoader(
            ds,
            shuffle=False,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn_partial,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for name, ds in valid_datasets
    }

    # determine training steps
    override_train_step = False
    if args.train_steps < 0:
        args.train_steps = (
            sum(len(v) for v in train_loaders.values()) * args.train_epochs
        )
        override_train_step = True

    accelerator.print(
        f"******************************** Training step before prepare: {args.train_steps} ********************************"
    )

    accelerator.print("loading model")
    if is_t5gemma2:
        model = F2LLMT5Gemma2(args.model_path, args.max_seq_length, args=args)
    else:
        model = F2LLM(args.model_path, args.max_seq_length, args=args)
    model.lm.gradient_checkpointing_enable()
    # set seed again to make sure that different models share the same seed
    set_seed(0)

    optimizer = AdamW(
        model.lm.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.98),
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    AcceleratorState().deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = args.train_batch_size

    accelerator.print("preparing model")
    model.lm, optimizer, lr_scheduler = accelerator.prepare(
        model.lm, optimizer, lr_scheduler
    )
    model.set_device()
    train_dataloader = MultiLoader(train_loaders, accelerator)
    for k, v in valid_loaders.items():
        valid_loaders[k] = accelerator.prepare(v)

    # if training on multiple GPUs, length of dataloader would have changed
    if override_train_step:
        args.train_steps = len(train_dataloader) * args.train_epochs
    accelerator.print(
        f"******************************** Training step after prepare: {args.train_steps} ********************************"
    )

    # ------------------------------------------------------------------
    # Build MTEB evaluator (only when eval_steps > 0).
    # evaluate_retrieval internally calls dist.get_rank() / dist.get_world_size()
    # which work because accelerate with DeepSpeed initialises torch.distributed.
    # ------------------------------------------------------------------
    eval_tasks = get_eval_tasks(args.eval_set)
    # Evaluator settings depend on whether the model is a causal LM (qwen-style,
    # last-token pooling, no special tokens added by tokenizer) or a bidirectional
    # encoder (T5Gemma2-style, mean pooling, tokenizer adds BOS/EOS itself).
    if is_t5gemma2:
        _eval_instruction_template = instruction_template_embeddinggemma
        _eval_add_special_tokens = True
        _eval_eot_id = None
    else:
        _eval_instruction_template = instruction_template_qwen3
        _eval_add_special_tokens = False
        _eval_eot_id = tokenizer.pad_token_id

    evaluator = EvaluateRetrieval(
        tasks=eval_tasks,
        tokenizer=tokenizer,
        instruction_template=_eval_instruction_template,
        padding_side="right",
        add_special_tokens=_eval_add_special_tokens,
        eot_id=_eval_eot_id,
        max_samples=1_000_000,
    )

    accelerator.print("start training")

    accelerate_train(
        args,
        accelerator,
        model,
        train_dataloader,
        valid_loaders,
        optimizer,
        lr_scheduler,
        sum(len(d[1]) for d in train_datasets),
        evaluator=evaluator,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )


if __name__ == "__main__":
    main()
