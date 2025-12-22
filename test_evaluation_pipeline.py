from utils.test_retrieval import evaluate_retrieval
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from utils._create_dataloaders import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)

model = SentenceTransformer("google/embeddinggemma-300m")

model.prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B",
        use_fast=False,
        trust_remote_code=True,
    )

    retrieval_evaluator = evaluate_retrieval(
        tasks=["ArguAna"],
        tokenizer=tokenizer,
        instruction_template=instruction_template_qwen3,
    )

    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B",
        dtype=torch.bfloat16,
    ).to("cuda")

    model = DDP(model, device_ids=[LOCAL_RANK])
    results = retrieval_evaluator.evaluate(model, batch_size=32)
    print(results)


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
