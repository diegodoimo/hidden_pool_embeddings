from inference.hard_negative_mining import HardNegativesMiner
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from inference.create_datasets import instruction_template_qwen3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()
    return args


path_to_name = {"Qwen/Qwen3-Embedding-0.6B": "qwen3_600m", 
            "Qwen/Qwen3-Embedding-8B": "qwen3_8b"}


def main():
    args = parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    miner = HardNegativesMiner(
        path=f"./results/datasets_negatives/{path_to_name[args.model_name_or_path]}",
        tasks=["nfcorpus"], #msmarco
        tokenizer=tokenizer,
        instruction_template=instruction_template_qwen3,
        padding_side="right",
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")


    model = DDP(model, device_ids=[LOCAL_RANK])

    miner.mine_negatives(model, batch_size=32)

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
