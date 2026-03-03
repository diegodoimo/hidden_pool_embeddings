import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum number of tokens per sequence.",
    )
    parser.add_argument(
        "--length_strategy",
        type=str,
        default="none",
        choices=["none", "truncate", "filter"],
        help=(
            "How to enforce max_seq_len. "
            "'truncate': truncate at the tokenizer level in the collate function. "
            "'filter': remove dataset rows where any component (query, positive, or a negative) "
            "exceeds max_seq_len tokens. "
            "'none': no enforcement."
        ),
    )

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_min_fact", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_thresh", type=float, default=1.0)
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--checkpointing_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=10)

    parser.add_argument("--measure_baselines", action="store_true", help="")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--out_filename", type=str, default="")
    parser.add_argument("--save_checkpoint", action="store_true")

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--dist_type", type=str, default="ddp", help="Distributed backend: ddp or fsdp"
    )

    parser.add_argument("--distributed_loss", action="store_true")
    parser.add_argument("--attention_pooling", action="store_true")
    parser.add_argument("--attention_dim", type=int, default=None)
    parser.add_argument("--cls_query_pooling", action="store_true")

    parser.add_argument(
        "--negatives_dir",
        type=str,
        default=None,
        help="Path to directory containing hard negative datasets (e.g. results/datasets_negatives/qwen3_600m)",
    )
    parser.add_argument(
        "--batch_strategy",
        type=str,
        default="sequential",
        choices=["mixed", "sequential", "grouped"],
        help="Batching strategy: 'mixed' (default, standard DistributedSampler), "
        "'sequential' (process one dataset at a time), "
        "'grouped' (interleave datasets round-robin, each batch from one dataset).",
    )
    parser.add_argument("--num_hard_negatives", type=int, default=8)
    parser.add_argument(
        "--eval_set",
        type=str,
        default="mteb_eng_v2",
        choices=["mteb_multilingual_v2", "mteb_eng_v2", "mteb_eng_v2_reduced"],
        help="MTEB evaluation set: full eng/multilingual or reduced task subset",
    )
    parser.add_argument(
        "--eval_task_types",
        type=str,
        nargs="*",
        default=None,
        help="Restrict evaluation to specific task types (e.g. Retrieval STS). "
        "If omitted, all tasks in the eval_set are used.",
    )
    parser.add_argument(
        "--train_subset",
        type=str,
        default="full",
        choices=["full", "reduced"],
        help="MTEB evaluation set: full eng/multilingual or reduced task subset",
    )
    parser.add_argument(
        "--tokenize_dataset",
        action="store_true",
        help=(
            "Pre-tokenize the dataset at construction time using "
            "create_pretokenized_hard_negatives_datasets.  The collate function "
            "will then only pad pre-computed token IDs, avoiding any tokenizer "
            "calls at training time."
        ),
    )

    args = parser.parse_args()
    return args
