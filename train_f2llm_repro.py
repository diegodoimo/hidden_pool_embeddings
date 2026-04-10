from f2llm_repro.f2llm_train import (
    accelerate_train,
    load_training_state,
    CLASSIFICATION_DATASETS,
    RETRIEVAL_DATASETS,
    EmbeddingModelEvalWrapper,
)
from f2llm_repro.model import F2LLM, F2LLMT5Gemma2, F2LLMT5Gemma2Decoder
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
    instruction_template_f2llm,
)
from torch.utils.data import RandomSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """

    def __init__(self, loader_dict, accelerator, batch_recorder=None):
        self.loader_dict = loader_dict
        self.batch_recorder = batch_recorder
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
                if self.batch_recorder is not None:
                    self.batch_recorder.record(batch)
                yield batch
            except StopIteration:
                idx = self.names.index(name)
                self.names.pop(idx)  # this dataset has no batch left
                self.weights.pop(idx)


class BatchMetadataRecorder:
    """Buffered writer for per-batch sample metadata.

    Writes one JSON object per line with:
    - batch_id
    - data_name
    - rank
    - data_index (list for the batch)
    """

    def __init__(self, output_dir, rank, run_label="", flush_every=200):
        self.output_dir = output_dir
        self.rank = int(rank)
        self.run_label = str(run_label).strip()
        self.flush_every = max(1, int(flush_every))
        self.buffer = []
        self.batch_id = 0

        os.makedirs(self.output_dir, exist_ok=True)
        filename = (
            f"batch_sample_map_{self.run_label}_rank{self.rank}.jsonl"
            if self.run_label
            else f"batch_sample_map_rank{self.rank}.jsonl"
        )
        self.file_path = os.path.join(self.output_dir, filename)
        self._fh = None

    def record(self, batch):
        data_indices = batch.get("data_indices", None)
        if data_indices is None:
            return
        if isinstance(data_indices, torch.Tensor):
            data_indices = data_indices.detach().cpu().tolist()

        self.buffer.append(
            {
                "batch_id": self.batch_id,
                "data_name": batch.get("dataset_name", None),
                "rank": self.rank,
                "data_index": data_indices,
            }
        )
        self.batch_id += 1

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        if self._fh is None:
            self._fh = open(self.file_path, "w")
        self._fh.write("\n".join(json.dumps(rec) for rec in self.buffer) + "\n")
        self._fh.flush()
        self.buffer = []

    def close(self):
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# def _stack(input_ids, max_len, tokenizer):
#     """OLD: sum(data, []) is O(n**2) — it repeatedly copies growing lists.
#     With 432 sequences of ~500 tokens each that's ~216K tokens copied
#     quadratically."""
#     data = [ids[:max_len] for ids in input_ids]
#     data = [
#         ids if ids[-1] == tokenizer.eos_token_id
#         else ids[:-1] + [tokenizer.eos_token_id]
#         for ids in data
#     ]
#     lens = [len(x) for x in data]
#     tensor = torch.tensor(sum(data, []))  # O(n**2) list concat
#     return tensor.split(lens)

def _stack(input_ids, max_len, tokenizer):
    """Truncate, ensure EOS, pack into a flat tensor and split.

    Uses itertools.chain instead of sum(lists, []) to avoid O(n**2)
    list concatenation.
    """
    from itertools import chain

    data = [ids[:max_len] for ids in input_ids]
    data = [
        (
            ids
            if ids[-1] == tokenizer.eos_token_id
            else ids[:-1] + [tokenizer.eos_token_id]
        )
        for ids in data
    ]
    lens = [len(x) for x in data]
    tensor = torch.tensor(list(chain.from_iterable(data)))  # O(n) flat concat
    return tensor.split(lens)


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
        "data_indices": torch.tensor(
            [int(sample["data_index"]) for sample in batch_raw], dtype=torch.long
        ),
    }


def collate_fn2(batch_raw, args, _stack, tokenizer, classification_datasets):
    """Variant of ``collate_fn`` for datasets pre-annotated with integer doc IDs.

    Expected dataset fields (in addition to ``query_input_ids`` and
    ``passage_input_ids``):

    * ``negative_token_ids``  – list of token-id lists, one per hard negative
      (all candidates stored together rather than as separate named fields).
    * ``positive_doc_id``     – int, document ID of the positive passage.
    * ``negative_doc_id``     – list[int], document IDs parallel to
      ``negative_input_ids``.

    The returned batch is identical to ``collate_fn`` plus two extra keys:

    * ``positive_doc_ids``  – int32 tensor, shape [bs].
    * ``negative_doc_ids``  – int32 tensor, shape [bs, num_hard_neg],
      aligned with the selected hard negatives.
    """
    num_hard_neg = (
        1
        if batch_raw[0]["dataset_name"] in classification_datasets
        else args.num_hard_neg
    )

    # Sample from however many hard negatives are available (flexible pool size).
    num_available = len(batch_raw[0]["negative_token_ids"])
    hard_neg_indices = (
        [0] if num_hard_neg == 1 else random.sample(range(num_available), num_hard_neg)
    )

    input_ids = _stack(
        [s["query_token_ids"] for s in batch_raw]
        + [s["positive_token_ids"] for s in batch_raw]
        + [s["negative_token_ids"][i] for s in batch_raw for i in hard_neg_indices],
        args.max_seq_length,
        tokenizer,
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    positive_doc_ids = torch.tensor(
        [int(s["positive_doc_id"]) for s in batch_raw], dtype=torch.int32
    )
    negative_doc_ids = torch.tensor(
        [[int(s["negative_doc_id"][i]) for i in hard_neg_indices] for s in batch_raw],
        dtype=torch.int32,
    )

    return {
        "input_ids": input_ids,
        "seq_lens": seqlens,
        "attention_mask": attention_masks,
        "bs": len(batch_raw),
        "dataset_name": batch_raw[0]["dataset_name"],
        "data_indices": torch.tensor(
            [int(sample["data_index"]) for sample in batch_raw], dtype=torch.long
        ),
        "positive_doc_ids": positive_doc_ids,
        "negative_doc_ids": negative_doc_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tb_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
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
    parser.add_argument("--measure_baselines", action="store_true")
    parser.add_argument("--only_retrieval", action="store_true")
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
    parser.add_argument(
        "--out_filename",
        type=str,
        default="",
        help="Optional label appended to the output log file: train_logs_<name>.json. Defaults to train_logs.json when empty.",
    )
    parser.add_argument(
        "--batch_map_flush_every",
        type=int,
        default=200,
        help="Flush batch metadata to disk every N batches per rank.",
    )
    parser.add_argument(
        "--attention_pooling",
        action="store_true",
        help="Use gated-attention pooling over per-layer hidden states "
        "(EmbeddingT5Gemma2HiddenPool) instead of plain mean pooling.",
    )
    parser.add_argument(
        "--cls_query_pooling",
        action="store_true",
        help="Use a learnable CLS query token with gated-attention pooling "
        "(EmbeddingT5Gemma2HiddenPoolCLS). Implies --attention_pooling.",
    )
    parser.add_argument(
        "--num_pooling_heads",
        type=int,
        default=None,
        help="Number of attention heads for the layer-pooling mechanism. "
        "Defaults to num_hidden_layers + 1 (one head per layer representation). "
        "hidden_size must be divisible by this value.",
    )
    parser.add_argument(
        "--gated_attention",
        action="store_true",
        help="Use Qwen3-style headwise-gated attention pooling instead of "
        "standard multi-head self-attention pooling over layer representations.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the LLM encoder backbone and train only the pooling "
        "head (GatedAttention + Projection). Requires --attention_pooling.",
    )
    parser.add_argument(
        "--lora_cls",
        action="store_true",
        help="Use per-layer low-rank cross-attention adapters to compute a "
        "CLS representation from frozen backbone hidden states. The encoder "
        "is automatically frozen; only the adapters, CLS queries, "
        "GatedAttention, and Projection are trained.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="Rank of the low-rank cross-attention adapters used with "
        "--lora_cls. Default: 64.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention kernel: 'eager' (full O(L^2) materialisation), "
        "'sdpa' (fused PyTorch SDPA), 'flash_attention_2' (FlashAttn2). "
        "Default: 'sdpa'.",
    )
    parser.add_argument(
        "--cross_attention",
        action="store_true",
        help="Embed MLLama-style cross-attention layers within the T5Gemma2 "
        "encoder.  A learnable CLS token queries the frozen backbone's "
        "residual stream at selected layers via tanh-gated cross-attention "
        "blocks.  The backbone is automatically frozen; only the "
        "cross-attention layers, CLS query, and Projection are trained.",
    )
    parser.add_argument(
        "--lora_cls_attn",
        action="store_true",
        help="Embed a learnable CLS token in the self-attention of each "
        "encoder layer with LoRA adapters on Q/O (masked to CLS position). "
        "An asymmetric attention mask keeps the backbone frozen for non-CLS "
        "tokens.  CLS states from all layers are pooled via GatedAttention.",
    )
    parser.add_argument(
        "--cross_attention_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices (0-based) at which to place cross-attention "
        "blocks.  Default (when --cross_attention is set): [3, 8, 13, 18, 23] "
        "(every ~5 layers, following the MLLama spacing for 26 encoder layers).",
    )
    parser.add_argument(
        "--use_decoder",
        action="store_true",
        help="Use the T5Gemma2 *decoder* (causal, EOS-pooling) instead of the "
        "encoder (bidirectional, mean-pooling). Only applies when model_path "
        "contains 't5gemma-2'.",
    )
    parser.add_argument(
        "--max_eval_queries",
        type=int,
        default=200000,
        help="Cap the number of queries evaluated per task (random subsample). "
        "Useful for large reranking tasks like MindSmallReranking (~2.4M queries).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory saved by save_training_state "
        "(e.g. output_dir/checkpoint_epoch_2). Resumes model, optimizer, "
        "scheduler, and training loop from that point.",
    )
    args = parser.parse_args()

    if args.cls_query_pooling:
        args.attention_pooling = True

    # Default cross-attention layer positions (MLLama-style spacing for 26 layers)
    if args.cross_attention and args.cross_attention_layers is None:
        args.cross_attention_layers = [3, 8, 13, 18, 23]

    # Resolve default num_pooling_heads from the model config
    if args.num_pooling_heads is None and args.attention_pooling:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_path)
        text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "encoder", cfg)
        args.num_pooling_heads = text_cfg.num_attention_heads

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
    is_t5gemma2_decoder = is_t5gemma2 and getattr(args, "use_decoder", False)

    # ---- Validate training-mode flag combinations ----
    if args.lora_cls_attn and args.cross_attention:
        raise ValueError(
            "--lora_cls_attn and --cross_attention are mutually exclusive."
        )
    if args.lora_cls_attn and args.lora_cls:
        raise ValueError(
            "--lora_cls_attn and --lora_cls are mutually exclusive."
        )
    if args.lora_cls_attn and args.attention_pooling:
        raise ValueError(
            "--lora_cls_attn already includes GatedAttention pooling; "
            "do not combine with --attention_pooling."
        )
    if args.lora_cls_attn and args.freeze_backbone:
        raise ValueError(
            "--lora_cls_attn already freezes the backbone; "
            "do not combine with --freeze_backbone."
        )
    if args.cross_attention and args.lora_cls:
        raise ValueError(
            "--cross_attention and --lora_cls are mutually exclusive."
        )
    if args.cross_attention and args.attention_pooling:
        raise ValueError(
            "--cross_attention already includes GatedAttention pooling over "
            "cross-attention CLS states; do not combine with --attention_pooling."
        )
    if args.cross_attention and args.freeze_backbone:
        raise ValueError(
            "--cross_attention and --freeze_backbone are mutually exclusive. "
            "--cross_attention already freezes the backbone."
        )
    if args.lora_cls and args.freeze_backbone:
        raise ValueError(
            "--lora_cls and --freeze_backbone are mutually exclusive. "
            "--lora_cls already freezes the backbone and trains its own "
            "cross-attention adapters."
        )
    if args.lora_cls and args.attention_pooling:
        raise ValueError(
            "--lora_cls is incompatible with --attention_pooling. "
            "--lora_cls builds its own pooling head; do not combine them."
        )
    if args.lora_cls and args.cls_query_pooling:
        raise ValueError(
            "--lora_cls is incompatible with --cls_query_pooling. "
            "--lora_cls builds its own CLS mechanism; do not combine them."
        )
    if args.freeze_backbone and not args.attention_pooling:
        raise ValueError(
            "--freeze_backbone requires --attention_pooling. "
            "Without a pooling head there are no trainable parameters."
        )
    if args.cls_query_pooling and not args.attention_pooling:
        raise ValueError(
            "--cls_query_pooling requires --attention_pooling."
        )
    if not is_t5gemma2 and (args.lora_cls or args.freeze_backbone):
        raise ValueError(
            "--lora_cls and --freeze_backbone are only supported for "
            "T5Gemma2 models (encoder or decoder)."
        )

    # Hard-coded cache directories per model family (overridable via --cache_dir).
    if args.cache_dir is None:
        if is_t5gemma2:
            args.cache_dir = "./f2llm_repro/cache/f2llm-prompt_t5gemma-2"
        else:
            args.cache_dir = "./f2llm_repro/cache/f2llm-prompt_qwen3"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    set_seed(0)
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    #train_datasets, valid_datasets = [], []
    train_datasets = []
    accelerator.print("loading datasets")

    tokenized_folder_suffix = "f2llm-prompt_qwen3-tok"
    if "t5gemma-2" in args.model_path.lower():
        tokenized_folder_suffix = "f2llm-prompt_t5gemma-2-tok"
        
    args.train_data_path = os.path.join(args.train_data_path, tokenized_folder_suffix)
    # NOTE: main_process_first() was removed because the NCCL barrier it
    # creates can time out (default 30 min) when rank 0 sequentially loads
    # many datasets from CephFS.  All ranks now load in parallel; HuggingFace
    # datasets handles concurrent cache creation via file locking.
    #
    # with accelerator.main_process_first():
    #     for f in sorted(
    #         f for f in os.listdir(args.train_data_path) if f.endswith(".parquet")
    #     ):
    #         dataset_name = f.split(".parquet")[0]
    #         if dataset_name not in RETRIEVAL_DATASETS and args.only_retrieval:
    #             continue
    #         accelerator.print(f"loading {dataset_name}")
    #         dataset = load_dataset(
    #             "parquet",
    #             data_files=os.path.join(args.train_data_path, f),
    #             cache_dir=args.cache_dir,
    #         )["train"]
    #         dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    #         dataset = dataset.map(
    #             lambda _, idx: {"data_index": idx},
    #             with_indices=True,
    #             desc=f"adding data_index to {dataset_name}",
    #         )
    #         train_datasets.append((dataset_name, dataset))

    for f in sorted(
        f for f in os.listdir(args.train_data_path) if f.endswith(".parquet")
    ):
        dataset_name = f.split(".parquet")[0]
        if dataset_name not in RETRIEVAL_DATASETS and args.only_retrieval:
            continue

        accelerator.print(f"loading {dataset_name}")

        dataset = load_dataset(
            "parquet",
            data_files=os.path.join(args.train_data_path, f),
            cache_dir=args.cache_dir,
        )["train"]

        dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
        dataset = dataset.map(
            lambda _, idx: {"data_index": idx},
            with_indices=True,
            desc=f"adding data_index to {dataset_name}",
        )
        train_datasets.append((dataset_name, dataset))
            # dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
            # train_datasets.append((dataset_name, dataset["train"]))
            # valid_datasets.append((dataset_name, dataset["test"]))

    collate_fn_partial = partial(
        collate_fn2,
        args=args,
        _stack=_stack,
        tokenizer=tokenizer,
        classification_datasets=CLASSIFICATION_DATASETS,
    )

    train_loaders = {
        # shuffle=True,
        name: DataLoader(
            ds,
            sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(0)),
            batch_size=args.train_batch_size,
            collate_fn=collate_fn_partial,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for name, ds in train_datasets
    }

    # determine training steps
    override_train_step = False
    if args.train_steps < 0:
        args.train_steps = (
            sum(len(v) for v in train_loaders.values()) * args.train_epochs
        )
        override_train_step = True

    # Pooling label: gated or standard attention, with number of heads
    def _pooling_suffix():
        prefix = "_gated_attn" if args.gated_attention else "_attn_pooling"
        heads = args.num_pooling_heads
        return f"{prefix}_h{heads}" if heads is not None else prefix

    if is_t5gemma2_decoder:
        model_name = "t5gemma2_decoder"
        if args.lora_cls:
            model_name += f"_lora_cls_r{args.lora_rank}"
        elif args.attention_pooling:
            model_name += _pooling_suffix()
            if args.freeze_backbone:
                model_name += "_frozen"
    elif is_t5gemma2:
        model_name = "t5gemma2"
        if args.lora_cls_attn:
            model_name += f"_lora_cls_attn_r{args.lora_rank}"
        elif args.cross_attention:
            layers_str = "_".join(str(i) for i in args.cross_attention_layers)
            model_name += f"_cross_attn_L{layers_str}"
        elif args.lora_cls:
            model_name += f"_lora_cls_r{args.lora_rank}"
        elif args.attention_pooling:
            model_name += _pooling_suffix()
            if args.cls_query_pooling:
                model_name += "_cls_query"
            if args.freeze_backbone:
                model_name += "_frozen"
    elif "qwen3" in args.model_path.lower():
        model_name = "qwen3"
    else:
        model_name = "model"

    suffix = (
        f"deepspeed_{model_name}"
        f"_gpus{args.num_processes}"
        f"_bs{args.train_batch_size * args.num_processes}"
        f"_lr{args.learning_rate}"
        f"_wd{args.weight_decay}"
    )
    if args.out_filename:
        args.out_filename = f"{args.out_filename}_{suffix}"
    else:
        args.out_filename = suffix

    accelerator.print(
        f"******************************** Training step before prepare: {args.train_steps} ********************************"
    )

    accelerator.print("loading model")
    if is_t5gemma2_decoder:
        model = F2LLMT5Gemma2Decoder(args.model_path, args.max_seq_length, args=args)
        if args.lora_cls:
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            accelerator.print(
                f"[lora_cls] decoder frozen — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.freeze_backbone:
            model.lm.encoder.gradient_checkpointing_enable()
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            accelerator.print(
                f"[freeze_backbone] decoder frozen — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        else:
            model.lm.encoder.gradient_checkpointing_enable()
    elif is_t5gemma2:
        model = F2LLMT5Gemma2(args.model_path, args.max_seq_length, args=args)
        if args.lora_cls_attn:
            # LoRA CLS in-attention: freeze entire encoder backbone.
            # LoRA adapters, CLS query, pooling, projection are in the wrapper
            # (outside encoder) and stay trainable.  The lora_layers hold
            # references to frozen encoder layers but own their own LoRA params.
            model.lm.encoder.gradient_checkpointing_enable()
            # Enable gradient checkpointing on LoRA wrapper layers too —
            # they live outside the encoder so encoder.gradient_checkpointing_enable()
            # doesn't reach them.
            gc_func = model.lm.encoder.layers[0]._gradient_checkpointing_func
            for lora_layer in model.lm.lora_layers:
                lora_layer._gradient_checkpointing_func = gc_func
                lora_layer.gradient_checkpointing = True
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            accelerator.print(
                f"[lora_cls_attn] backbone frozen, LoRA + pooling trainable — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.cross_attention:
            # Cross-attention: freeze backbone, keep cross-attn layers trainable.
            # Pooling, Projection (in wrapper, outside encoder) are already unfrozen.
            model.lm.encoder.gradient_checkpointing_enable()
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            # Unfreeze cross-attention layers and CLS query inside the encoder
            model.lm.encoder.cls_query.requires_grad = True
            for p in model.lm.encoder.cross_attn_layers.parameters():
                p.requires_grad = True
            accelerator.print(
                f"[cross_attention] backbone frozen, cross-attn + pooling trainable — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.lora_cls:
            # LoRA CLS: encoder is frozen inside the model (torch.no_grad),
            # gradient checkpointing is not needed on the frozen encoder.
            # Explicitly freeze encoder parameters so DeepSpeed doesn't
            # allocate optimizer states for them.
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            accelerator.print(
                f"[lora_cls] encoder frozen — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.freeze_backbone:
            # Freeze encoder, train only pooling + projection head.
            model.lm.encoder.gradient_checkpointing_enable()
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            accelerator.print(
                f"[freeze_backbone] encoder frozen — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        else:
            model.lm.encoder.gradient_checkpointing_enable()
    else:
        model = F2LLM(args.model_path, args.max_seq_length, args=args)
        model.lm.gradient_checkpointing_enable()
    # set seed again to make sure that different models share the same seed
    set_seed(0)

    # Only pass trainable parameters to the optimizer.
    trainable_params = [p for p in model.lm.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
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

    batch_recorder = BatchMetadataRecorder(
        output_dir=os.path.join(args.output_dir, "batch_sample_map"),
        rank=accelerator.process_index,
        run_label=args.out_filename,
        flush_every=args.batch_map_flush_every,
    )
    train_dataloader = MultiLoader(
        train_loaders, accelerator, batch_recorder=batch_recorder
    )

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
    task_types = (
        ["Reranking", "Retrieval", "STS", "Summarization"]
        if args.only_retrieval
        else None
    )
    eval_tasks = get_eval_tasks(args.eval_set, task_types)
    # Evaluator settings: T5Gemma2 (encoder or decoder) uses a tokenizer that
    # adds BOS/EOS automatically; Qwen-style models use manual EOS appending.

    _eval_instruction_template = instruction_template_f2llm
    if is_t5gemma2:
        # _eval_instruction_template = instruction_template_embeddinggemma
        _eval_add_special_tokens = True
        _eval_eot_id = None
    else:
        # _eval_instruction_template = instruction_template_qwen3
        _eval_add_special_tokens = False
        # in qwen embedding is the pad_token
        _eval_eot_id = tokenizer.eos_token_id

    evaluator = EvaluateRetrieval(
        tasks=eval_tasks,
        tokenizer=tokenizer,
        instruction_template=_eval_instruction_template,
        padding_side="right",
        add_special_tokens=_eval_add_special_tokens,
        eot_id=_eval_eot_id,
        max_samples=1_000_000,
        max_eval_queries=args.max_eval_queries,
    )

    start_epoch, start_step = 0, 0
    if args.resume_from_checkpoint:
        start_epoch, start_step = load_training_state(
            accelerator, args.resume_from_checkpoint
        )

    accelerator.print("start training")

    try:
        accelerate_train(
            args,
            accelerator,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            sum(len(d[1]) for d in train_datasets),
            evaluator=evaluator,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            eval_wrapper_class=EmbeddingModelEvalWrapper if (is_t5gemma2 or is_t5gemma2_decoder) else None,
            start_epoch=start_epoch,
            start_step=start_step,
        )
    finally:
        batch_recorder.close()


if __name__ == "__main__":
    main()
