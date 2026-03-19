from inference.mine_hard_negatives import HardNegativesMiner
import os
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from utils.create_datasets import instruction_template_qwen3
from datetime import timedelta
from tasks.helpers import validate_and_select_tasks
from utils.helpers import print_memory_consumed
from models.modules import add_pooling_layers, last_token_pool


path_to_name = {
    "Qwen/Qwen3-Embedding-0.6B": "qwen3_600m",
    "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific task names to mine hard negatives for (e.g., 'msmarco' 'hotpotqa'). Takes precedence over --task_types.",
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=None,
        help="Select task types to mine hard negatives for. Can specify multiple types. Ignored if --task_names is provided.",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterative_encode_threshold", type=int, default=10**7)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])

    dist.init_process_group(
        "nccl",
        device_id=LOCAL_RANK,
        timeout=timedelta(seconds=1800),
    )
    torch.cuda.set_device(dist.get_rank())

    # enable tensorfloat32
    torch.set_float32_matmul_precision("high")

    # Select tasks based on task_names (if provided) or task_types
    selected_tasks = validate_and_select_tasks(args.task_names, args.task_types)

    if RANK == 0:
        if args.task_names is not None:
            print(f"Selected specific tasks: {args.task_names}")
        else:
            print(f"Selected task types: {args.task_types}")
        print(f"Tasks to process: {selected_tasks}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    max_length = min(args.max_length, model.config.max_position_embeddings)

    miner = HardNegativesMiner(
        path=f"./results/datasets_negatives/{path_to_name[args.model_name_or_path]}",
        model_name=path_to_name[args.model_name_or_path],
        task_names=selected_tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template_qwen3,
        padding_side="right",
        max_length=max_length,
        add_special_tokens=False,
        eot_id=tokenizer.pad_token_id,
        iterative_encode_threshold=args.iterative_encode_threshold,
    )

    if RANK == 0:
        print("model loaded")
    dist.barrier()
    model = model.eval()
    # ddp is only needed for training here we are adding gradient buffers and the memory occupied with doubl
    # model = DDP(model, device_ids=[dist.get_rank()])
    model = torch.compile(model)
    model = add_pooling_layers(model, pool_fn=last_token_pool)

    if RANK == 0:
        print("model wrapped in DDP and compile")
        print_memory_consumed()
    dist.barrier()

    miner.mine_negatives(model, batch_size=args.batch_size)
    dist.destroy_process_group()


if __name__ == "__main__":

    main()
