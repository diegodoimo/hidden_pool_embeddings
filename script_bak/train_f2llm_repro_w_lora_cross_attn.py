from f2llm_repro.f2llm_train import (
    accelerate_train,
    load_training_state,
    CLASSIFICATION_DATASETS,
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
from tqdm import tqdm

from inference.test_retrieval_ddp_update import evaluate_retrieval as EvaluateRetrieval
from utils.create_datasets import (
    get_eval_tasks,
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
    instruction_template_f2llm,
)
from tasks import TRANSLATE_F2LLM_NAME, NAME_TO_TASK_TYPE
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
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=None)
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
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=[
            "Retrieval",
            "Reranking",
            "STS",
            "Summarization",
            "PairClassification",
            "Classification",
            "Clustering",
        ],
        help="Train only on datasets belonging to this MTEB task type. "
        "When omitted, all available datasets are used.",
    )
    parser.add_argument("--num_processes", type=int, default=0)
    parser.add_argument(
        "--eval_set",
        type=str,
        default="mteb_retrieval_subset",
        help="Name of the eval task set passed to get_eval_tasks() for mid-training MTEB evals.",
    )
    parser.add_argument(
        "--final_eval_set",
        type=str,
        default=None,
        help="Eval task set used for end-of-epoch evaluation. "
        "Defaults to --eval_set for two-stage stage 1, "
        "mteb_eng_v2_full otherwise.",
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
        "--pooling_mode",
        type=str,
        default="mean",
        choices=["both", "cls", "mean"],
        help="Which per-layer representations to pool over when "
        "--attention_pooling is enabled: 'mean' (mean-pooled tokens, no CLS), "
        "'cls' (learnable CLS token per layer), 'both' (CLS + mean-pooled). "
        "Default: 'mean'.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qwen3-style headwise-gated attention pooling instead of "
        "standard multi-head self-attention pooling over layer representations.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="Rank of the low-rank adapters used with "
        "--lora_cls_attn. Default: 64.",
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
    # ---- Two-stage training ----
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help="Enable 2-stage training for attention pooling. Stage 1 freezes "
        "the encoder and trains only the CLS token + pooling head on a "
        "fraction of the data. Stage 2 loads the stage-1 model and does "
        "end-to-end finetuning on the remaining data.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which stage to run (1 or 2). Only used with --two_stage.",
    )
    parser.add_argument(
        "--stage1_fraction",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of each task's data used for stage 1 (default: 1/3). "
        "Stage 2 uses the remaining samples with no overlap.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for the stratified stage-1 / stage-2 data split. "
        "Ensures reproducible splits across runs.",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Explicit path to the stage-1 model weights (model.pt). "
        "When omitted (recommended), the path is auto-derived from "
        "--stage1_lr and --stage1_wd by reconstructing the stage-1 "
        "output directory name.",
    )
    parser.add_argument(
        "--stage1_lr",
        type=float,
        default=5e-4,
        help="Learning rate used in stage 1 (default: 5e-4). Used by "
        "stage 2 to reconstruct the stage-1 output directory and "
        "locate the checkpoint automatically.",
    )
    parser.add_argument(
        "--stage1_wd",
        type=float,
        default=0.0,
        help="Weight decay used in stage 1 (default: 0.0). Used by "
        "stage 2 to reconstruct the stage-1 output directory and "
        "locate the checkpoint automatically.",
    )
    parser.add_argument(
        "--stage2_head_lr",
        type=float,
        default=None,
        help="Separate learning rate for the pooling head (CLS, pooling, "
        "projection) during stage 2. When set, the encoder uses "
        "--learning_rate (default 2e-5) while the head uses this value "
        "(e.g. 1e-4). Only used with --two_stage --stage 2.",
    )
    parser.add_argument(
        "--cls_init",
        type=str,
        default="mean_embed",
        choices=["mean_embed", "random"],
        help="How to initialize the CLS query when it is absent from the "
        "stage-1 checkpoint (e.g. stage 1 used --pooling_mode mean and "
        "stage 2 uses cls/both). 'mean_embed' (default): mean of the "
        "encoder's token embedding table. 'random': keep the default "
        "random nn.Parameter initialization.",
    )
    # ---- Procrustes alignment ----
    parser.add_argument(
        "--procrustes_alignment",
        action="store_true",
        help="Insert a per-layer orthogonal alignment (Generalized Procrustes, "
        "arXiv:2602.06205) before the attention-pooling head. One orthogonal "
        "matrix Omega_k is learned per layer representation, applied as "
        "H_k @ Omega_k. Constraint enforced via torch.nn.utils.parametrizations.orthogonal.",
    )
    parser.add_argument(
        "--procrustes_init",
        type=str,
        default="identity",
        choices=["identity", "random"],
        help="How to initialise the orthogonal matrices. 'identity' (default): "
        "warm-start as no-op so finetuning starts from the un-aligned baseline. "
        "'random': random orthogonal init.",
    )
    parser.add_argument(
        "--procrustes_pretrain",
        action="store_true",
        help="Train ONLY the ProcrustesAlignment matrices using the GPA loss "
        "(sum_k ||H_k Omega_k - U||^2 with U the consensus over k). Encoder "
        "and pooling head are frozen and the encoder runs under no_grad. "
        "Use this before contrastive finetuning to warm-start the matrices, "
        "then point a follow-up finetuning run at the saved checkpoint via "
        "--procrustes_checkpoint.",
    )
    parser.add_argument(
        "--procrustes_lr",
        type=float,
        default=None,
        help="Learning rate used during --procrustes_pretrain. Defaults to "
        "--learning_rate when omitted.",
    )
    parser.add_argument(
        "--procrustes_checkpoint",
        type=str,
        default=None,
        help="Path to a model.pt (saved by a prior --procrustes_pretrain run) "
        "from which to load the ProcrustesAlignment weights into the current "
        "model. All other weights stay at their normal initialisation. Ignored "
        "when --procrustes_pretrain is set.",
    )
    parser.add_argument(
        "--stage1_pooling_mode",
        type=str,
        default="mean",
        choices=["both", "cls", "mean"],
        help="Pooling mode used in stage 1 (default: 'mean'). Used by "
        "stage 2 to reconstruct the stage-1 output directory and "
        "locate the checkpoint automatically.",
    )
    args = parser.parse_args()

    # Default cross-attention layer positions (MLLama-style spacing for 26 layers)
    if args.cross_attention and args.cross_attention_layers is None:
        args.cross_attention_layers = [3, 8, 13, 18, 23]

    # ---- Procrustes validation & implication ----
    if args.procrustes_pretrain:
        args.procrustes_alignment = True
    if args.procrustes_alignment and not (
        args.attention_pooling or args.lora_cls_attn or args.cross_attention
    ):
        parser.error(
            "--procrustes_alignment requires one of --attention_pooling, "
            "--lora_cls_attn, or --cross_attention (the heads that consume a "
            "per-view stack)."
        )

    # ---- Two-stage validation & LR defaults ----
    if args.two_stage:
        args.attention_pooling = True
        if not (0 < args.stage1_fraction <= 1):
            parser.error("--stage1_fraction must be in (0, 1]")
        # Apply stage-aware defaults when the user did not pass explicit values.
        # Stage 1 trains only the randomly-initialized pooling head (few
        # params, no pretrained weights to protect) → higher LR, no weight decay.
        # Stage 2 unfreezes the pretrained encoder for end-to-end finetuning →
        # lower LR, standard weight decay to avoid catastrophic forgetting.
        if args.learning_rate is None:
            args.learning_rate = 5e-4 if args.stage == 1 else 2e-5
        if args.weight_decay is None:
            args.weight_decay = 0.0 if args.stage == 1 else 1e-2

    # Default final_eval_set: reduced for stage 1, full otherwise.
    if args.final_eval_set is None:
        if args.two_stage and args.stage == 1:
            args.final_eval_set = args.eval_set
        else:
            args.final_eval_set = "mteb_eng_v2"

    # Build the set of f2llm parquet names that belong to the requested task type.
    # TRANSLATE_F2LLM_NAME maps f2llm_name → internal_name;
    # NAME_TO_TASK_TYPE maps internal_name → MTEB task type.
    if args.task_type:
        args._task_type_f2llm_names = sorted(
            f2llm_name
            for f2llm_name, internal_name in TRANSLATE_F2LLM_NAME.items()
            if NAME_TO_TASK_TYPE.get(internal_name) == args.task_type
        )

    # Fallback defaults when --two_stage is not used
    if args.learning_rate is None:
        args.learning_rate = 2e-5
    if args.weight_decay is None:
        args.weight_decay = 1e-2

    # Resolve default num_pooling_heads from the model config
    if args.num_pooling_heads is None and args.attention_pooling:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_path)
        text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "encoder", cfg)
        # For T5Gemma2EncoderConfig, num_attention_heads lives on .text_config
        text_cfg = getattr(text_cfg, "text_config", text_cfg)
        args.num_pooling_heads = text_cfg.num_attention_heads

    return args


def pretrain_procrustes(
    args, accelerator, model, train_dataloader, optimizer, lr_scheduler,
):
    """Train only ``model.lm.procrustes`` matrices with the GPA loss.

    For each batch:
      1. Encoder forward under ``no_grad`` to get the (B, K, D) per-view stack
         (via ``model.lm.compute_gpa_loss``).
      2. Rotate each view by its orthogonal matrix and compute
         ``sum_k ||H_k Omega_k - U||^2`` with U the consensus mean.
      3. Backprop into the Omega_k only.

    The encoder + pooling head must be frozen by the caller; the optimizer
    must already be restricted to the Procrustes parameters.
    """
    accelerator.print(
        "*** Procrustes pretrain — frozen encoder, GPA loss over per-view stack ***"
    )
    accelerator.print(f" Num epochs = {args.train_epochs}")
    accelerator.print(f" Per device batch size = {args.train_batch_size}")
    accelerator.print(f" Total training steps = {args.train_steps}")

    model.lm.train()
    filename = (
        "_" + args.out_filename if getattr(args, "out_filename", "") != "" else ""
    )
    stats = {"train": {}}
    pbar = tqdm(
        range(args.train_steps), disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    running_loss = torch.tensor(0.0, device=model.device)
    running_count = torch.tensor(0, device=model.device)

    for epoch in range(args.train_epochs):
        accelerator.print(f"*** procrustes epoch {epoch+1} ***")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            # Route through the engine forward so DeepSpeed sees the
            # parameter set; gpa_loss=True swaps the head for the GPA loss.
            loss = model.lm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                gpa_loss=True,
            )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if optimizer.param_groups[0]["lr"] < args.min_lr:
                for g in optimizer.param_groups:
                    g["lr"] = args.min_lr

            running_loss += loss.detach().float()
            running_count += 1
            completed_steps += 1

            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)
                gathered = accelerator.gather(running_loss).sum()
                count = accelerator.gather(running_count).sum().clamp(min=1)
                avg = (gathered / count).item()
                running_loss.zero_()
                running_count.zero_()
                if accelerator.is_main_process:
                    stats["train"][completed_steps] = avg
                    accelerator.print(
                        f"[procrustes] step {completed_steps}/{args.train_steps} "
                        f"gpa_loss={avg:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    with open(
                        os.path.join(args.output_dir, f"train_logs{filename}.json"),
                        "w",
                    ) as f:
                        json.dump(stats, f, indent=4)


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
    if args.lora_cls_attn and args.attention_pooling:
        raise ValueError(
            "--lora_cls_attn already includes GatedAttention pooling; "
            "do not combine with --attention_pooling."
        )
    if args.cross_attention and args.attention_pooling:
        raise ValueError(
            "--cross_attention already includes GatedAttention pooling over "
            "cross-attention CLS states; do not combine with --attention_pooling."
        )
    if args.two_stage and not is_t5gemma2:
        raise ValueError(
            "--two_stage is only supported for T5Gemma2 encoder models."
        )
    if args.two_stage and (args.lora_cls_attn or args.cross_attention):
        raise ValueError(
            "--two_stage is incompatible with --lora_cls_attn "
            "and --cross_attention. It manages its own pooling architecture."
        )

    # Hard-coded cache directories per model family (overridable via --cache_dir).
    if args.cache_dir is None:
        if is_t5gemma2:
            args.cache_dir = "./f2llm_repro/cache/f2llm-prompt_t5gemma-2"
        else:
            args.cache_dir = "./f2llm_repro/cache/f2llm-prompt_qwen3"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    set_seed(0)

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
    for f in sorted(
        f for f in os.listdir(args.train_data_path) if f.endswith(".parquet")
    ):
        dataset_name = f.split(".parquet")[0]
        if args.task_type and dataset_name not in args._task_type_f2llm_names:
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

    # ---- Two-stage: stratified split per task ----
    if args.two_stage:
        if args.stage1_fraction >= 1.0:
            # stage1_fraction=1.0: both stages use the full dataset.
            # Stage 1 trains the pooling head only (frozen encoder);
            # Stage 2 loads stage-1 weights and fine-tunes end-to-end.
            accelerator.print(
                f"[two_stage] Stage {args.stage}: using 100% of data "
                f"({sum(len(d[1]) for d in train_datasets):,} samples across "
                f"{len(train_datasets)} tasks)"
            )
        else:
            stage1_datasets, stage2_datasets = [], []
            for name, ds in train_datasets:
                split = ds.train_test_split(
                    train_size=args.stage1_fraction,
                    shuffle=True,
                    seed=args.split_seed,
                )
                stage1_datasets.append((name, split["train"]))
                stage2_datasets.append((name, split["test"]))

            if args.stage == 1:
                train_datasets = stage1_datasets
                accelerator.print(
                    f"[two_stage] Stage 1: using {args.stage1_fraction:.1%} of data "
                    f"({sum(len(d[1]) for d in train_datasets):,} samples across "
                    f"{len(train_datasets)} tasks)"
                )
            else:
                train_datasets = stage2_datasets
                accelerator.print(
                    f"[two_stage] Stage 2: using {1 - args.stage1_fraction:.1%} of data "
                    f"({sum(len(d[1]) for d in train_datasets):,} samples across "
                    f"{len(train_datasets)} tasks)"
                )

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

    def _two_stage_model_name(stage, lr, wd, pooling_mode=None):
        """Build the model_name component for a given two-stage run."""
        if pooling_mode is None:
            pooling_mode = args.pooling_mode
        frac_str = f"{args.stage1_fraction:.2f}".rstrip("0").rstrip(".")
        name = f"t5gemma2_two_stage_s{stage}_frac{frac_str}" + _pooling_suffix()
        name += f"_{pooling_mode}"
        tt_tag = f"_task_{args.task_type}" if args.task_type else ""
        return (
            f"{name}"
            f"{tt_tag}"
            f"_gpus{args.num_processes}"
            f"_bs{args.train_batch_size * args.num_processes}"
            f"_lr{lr}"
            f"_wd{wd}"
        )
    

    def _minimal_s1_name(stage, lr, wd, pooling_mode=None):
        if pooling_mode is None:
            pooling_mode = args.pooling_mode
        frac_str = f"{args.stage1_fraction:.2f}".rstrip("0").rstrip(".")
        name = f"s{stage}_frac{frac_str}" + _pooling_suffix()
        name += f"_{pooling_mode}"
        tt_tag = f"_task_{args.task_type}" if args.task_type else ""
        return (
            f"{name}"
            f"{tt_tag}"
            f"_lr{lr}"
            f"_wd{wd}"
        )

    if is_t5gemma2_decoder:
        model_name = "t5gemma2_decoder"
        if args.attention_pooling:
            model_name += _pooling_suffix()
    elif is_t5gemma2:
        model_name = "t5gemma2"
        if args.two_stage:
            frac_str = f"{args.stage1_fraction:.2f}".rstrip("0").rstrip(".")
            model_name += f"_two_stage_s{args.stage}_frac{frac_str}" + _pooling_suffix()
            model_name += f"_{args.pooling_mode}"
            if args.stage == 2:
                s1_name = _minimal_s1_name(1, args.stage1_lr, args.stage1_wd, pooling_mode=args.stage1_pooling_mode)
                model_name+=f"_{s1_name}"
        
        elif args.lora_cls_attn:
            model_name += f"_lora_cls_attn_r{args.lora_rank}"
        
        elif args.cross_attention:
            layers_str = "_".join(str(i) for i in args.cross_attention_layers)
            model_name += f"_cross_attn_L{layers_str}"
        
        elif args.attention_pooling:
            model_name += _pooling_suffix()
            model_name += f"_{args.pooling_mode}"
    
    elif "qwen3" in args.model_path.lower():
        model_name = "qwen3"
    else:
        model_name = "model"

    task_type_tag = f"_task_{args.task_type}" if args.task_type else ""
    suffix = (
        f"{model_name}"
        f"{task_type_tag}"
        f"_gpus{args.num_processes}"
        f"_bs{args.train_batch_size * args.num_processes}"
        f"_lr{args.learning_rate}"
        f"_wd{args.weight_decay}"
    )
    user_prefix = args.out_filename  # original user value (may be "")
    if user_prefix:
        args.out_filename = f"{user_prefix}_{suffix}"
    else:
        args.out_filename = suffix

    # Build intermediate path: task_specific/ when --task_type is set,
    # stage1/ or stage2/ when --two_stage is set.
    subdir_parts = []
    if args.task_type:
        subdir_parts.append("task_specific")
    if args.two_stage:
        subdir_parts.append(f"stage{args.stage}")
    intermediate = os.path.join(*subdir_parts) if subdir_parts else ""

    # Auto-derive stage-1 checkpoint path for stage 2.
    # Reconstructs the stage-1 directory name from stage1_lr/stage1_wd
    # and the shared config (pooling, fraction, gpus, bs).
    if args.two_stage and args.stage == 2 and args.stage1_checkpoint is None:
        s1_suffix = _two_stage_model_name(1, args.stage1_lr, args.stage1_wd, pooling_mode=args.stage1_pooling_mode)
        s1_dir_name = f"{user_prefix}_{s1_suffix}" if user_prefix else s1_suffix
        # Stage-1 checkpoint lives under stage1/, not stage2/
        s1_intermediate_parts = []
        if args.task_type:
            s1_intermediate_parts.append("task_specific")
        s1_intermediate_parts.append("stage1")
        s1_intermediate = os.path.join(*s1_intermediate_parts)
        args.stage1_checkpoint = os.path.join(
            args.output_dir, s1_intermediate, s1_dir_name, "model.pt"
        )

    args.output_dir = os.path.join(args.output_dir, intermediate, args.out_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    accelerator.print(
        f"******************************** Training step before prepare: {args.train_steps} ********************************"
    )

    accelerator.print("loading model")
    if is_t5gemma2_decoder:
        model = F2LLMT5Gemma2Decoder(args.model_path, args.max_seq_length, args=args)
        model.lm.encoder.gradient_checkpointing_enable()
    elif is_t5gemma2:
        model = F2LLMT5Gemma2(args.model_path, args.max_seq_length, args=args)
        # Optional warm-start of the Procrustes matrices from a prior
        # --procrustes_pretrain run. Skipped when re-pretraining.
        if args.procrustes_checkpoint and not args.procrustes_pretrain:
            accelerator.print(
                f"[procrustes] loading alignment weights from {args.procrustes_checkpoint}"
            )
            _proc_sd = torch.load(
                args.procrustes_checkpoint, map_location="cpu", weights_only=True,
            )
            _proc_only = {k: v for k, v in _proc_sd.items() if k.startswith("procrustes.")}
            if not _proc_only:
                raise RuntimeError(
                    f"No 'procrustes.*' keys in {args.procrustes_checkpoint}; "
                    "is this a procrustes-pretrain checkpoint?"
                )
            result = model.lm.load_state_dict(_proc_only, strict=False)
            if result.unexpected_keys:
                accelerator.print(
                    f"[procrustes] WARNING unexpected keys: {result.unexpected_keys}"
                )
            accelerator.print(
                f"[procrustes] loaded {len(_proc_only)} alignment tensors"
            )
        # Procrustes pretrain: freeze everything except the alignment matrices.
        # Runs first so it short-circuits the other branches (it carries its
        # own loss + training loop and ignores the contrastive head).
        if args.procrustes_pretrain:
            for p in model.lm.parameters():
                p.requires_grad = False
            for p in model.lm.procrustes.parameters():
                p.requires_grad = True
            # Encoder runs under no_grad inside compute_gpa_loss, so no
            # activation checkpointing is needed.
            accelerator.print(
                f"[procrustes_pretrain] only ProcrustesAlignment trainable — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.two_stage and args.stage == 1:
            # Stage 1: freeze encoder, train pooling head only
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            # With mean pooling no trainable parameter flows through the
            # encoder, so we can skip autograd graph construction entirely
            # (no need for gradient checkpointing either).
            # For cls/both, cls_query backprops through the encoder so we
            # still need the graph — use gradient checkpointing to save memory.
            if args.pooling_mode == "mean" and hasattr(model.lm, "encoder_no_grad"):
                model.lm.encoder_no_grad = True
                accelerator.print("[two_stage:s1] encoder running under torch.no_grad()")
            else:
                model.lm.encoder.gradient_checkpointing_enable()
            accelerator.print(
                f"[two_stage:s1] encoder frozen — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.two_stage and args.stage == 2:
            # Stage 2: load stage-1 weights, unfreeze encoder for e2e training
            accelerator.print(f"[two_stage:s2] loading stage-1 weights from {args.stage1_checkpoint}")
            state_dict = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
            # strict=False allows switching pooling mode between stages
            # (e.g. stage 1 used mean → no cls_query in checkpoint,
            #  stage 2 uses cls/both → cls_query present in model).
            result = model.lm.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                accelerator.print(
                    f"[two_stage:s2] WARNING unexpected keys in checkpoint: {result.unexpected_keys}"
                )
            # Initialize cls_query when absent from stage-1 checkpoint
            if hasattr(model.lm, "cls_query") and "cls_query" in result.missing_keys:
                if args.cls_init == "mean_embed":
                    with torch.no_grad():
                        mean_emb = model.lm.encoder.embed_tokens.weight.mean(dim=0)
                        model.lm.cls_query.copy_(mean_emb.reshape(1, 1, -1))
                    accelerator.print(
                        "[two_stage:s2] cls_query initialized from mean of embedding table"
                    )
                else:
                    accelerator.print(
                        "[two_stage:s2] cls_query kept at random initialization"
                    )
            elif result.missing_keys:
                accelerator.print(
                    f"[two_stage:s2] WARNING missing keys: {result.missing_keys}"
                )
            model.lm.encoder.gradient_checkpointing_enable()
            accelerator.print(
                f"[two_stage:s2] end-to-end training — "
                f"trainable params: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}"
            )
        elif args.lora_cls_attn:
            # LoRA CLS in-attention: freeze entire encoder backbone.
            model.lm.encoder.gradient_checkpointing_enable()
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
            model.lm.encoder.gradient_checkpointing_enable()
            for p in model.lm.encoder.parameters():
                p.requires_grad = False
            model.lm.encoder.cls_query.requires_grad = True
            for p in model.lm.encoder.cross_attn_layers.parameters():
                p.requires_grad = True
            accelerator.print(
                f"[cross_attention] backbone frozen, cross-attn + pooling trainable — "
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
    # Stage 2 with --stage2_head_lr uses differential LRs: a lower LR for the
    # pretrained encoder and a higher LR for the pooling head (CLS, pooling,
    # projection) which can tolerate faster updates.
    if args.two_stage and args.stage == 2 and args.stage2_head_lr is not None:
        encoder_ids = {id(p) for p in model.lm.encoder.parameters()}
        encoder_params = [p for p in model.lm.encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.lm.parameters()
                       if p.requires_grad and id(p) not in encoder_ids]
        accelerator.print(
            f"[two_stage:s2] differential LR — "
            f"encoder ({sum(p.numel() for p in encoder_params):,} params): {args.learning_rate}, "
            f"head ({sum(p.numel() for p in head_params):,} params): {args.stage2_head_lr}"
        )
        optimizer = AdamW(
            [
                {"params": encoder_params, "lr": args.learning_rate},
                {"params": head_params, "lr": args.stage2_head_lr},
            ],
            weight_decay=args.weight_decay,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
        )
    else:
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

    train_dataloader = MultiLoader(
        train_loaders, accelerator, batch_recorder=None
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
    _eval_task_types = [args.task_type] if args.task_type else None
    eval_tasks = get_eval_tasks(args.eval_set, task_types=_eval_task_types)
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

    if args.procrustes_pretrain:
        pretrain_procrustes(
            args, accelerator, model, train_dataloader, optimizer, lr_scheduler,
        )
    else:
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

    # ---- Save stage-1 model weights for stage 2 ----
    if args.two_stage and args.stage == 1:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model.lm)
            save_path = os.path.join(args.output_dir, "model.pt")
            torch.save(unwrapped.state_dict(), save_path)
            accelerator.print(f"[two_stage:s1] model weights saved to {save_path}")
        accelerator.wait_for_everyone()

    # ---- Save procrustes-pretrained model weights ----
    if args.procrustes_pretrain:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model.lm)
            save_path = os.path.join(args.output_dir, "model.pt")
            torch.save(unwrapped.state_dict(), save_path)
            accelerator.print(
                f"[procrustes_pretrain] model weights saved to {save_path}. "
                "Use --procrustes_checkpoint to load these into a finetuning run."
            )
        accelerator.wait_for_everyone()

    # ---- Cleanup distributed state to avoid ResourceTracker / NCCL warnings ----
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
