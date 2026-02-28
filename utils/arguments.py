from argparse import ArgumentParser
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--out_filename", type=str, default="")
    parser.add_argument("--save_checkpoint", action="store_true")

    parser.add_argument("--reduce_loss", type=str, default="sum")

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--deepspeed_stage", type=int, default=0)

    parser.add_argument("--distributed_loss", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--attention_pooling", action="store_true")
    parser.add_argument("--attention_dim", type=int, default=None)
    parser.add_argument("--cls_query_pooling", action="store_true")
    parser.add_argument("--joint_batch", action="store_true")

    parser.add_argument(
        "--negatives_dir",
        type=str,
        default=None,
        help="Path to directory containing hard negative datasets (e.g. results/datasets_negatives/qwen3_600m)",
    )
    parser.add_argument(
        "--datasets_subset",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of dataset names to restrict loading (e.g. retrieval/general_retrieval/msmarco). "
        "Use with contrastive_datasets.QWEN3_600M_10DATASET_SUBSET for a 10-dataset example.",
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
    parser.add_argument("--max_query_len", type=int, default=256)
    parser.add_argument("--max_passage_len", type=int, default=512)
    parser.add_argument(
        "--instruction_template",
        type=str,
        default="qwen3",
        choices=["qwen3", "embeddinggemma"],
    )
    parser.add_argument(
        "--eval_set",
        type=str,
        default="mteb_eng_v2_20",
        choices=["mteb_multilingual_v2", "mteb_eng_v2", "mteb_eng_v2_20"],
        help="MTEB evaluation set: full eng/multilingual or reduced 20-task subset",
    )

    args = parser.parse_args()
    return args
