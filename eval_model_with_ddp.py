from inference.test_retrieval_ddp_update import evaluate_retrieval
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from inference.create_datasets import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)
import mteb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default="ArguAna")
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist.init_process_group(
        "nccl",
        device_id=LOCAL_RANK,
        timeout=timedelta(
            seconds=60
        ),  
    )
    torch.cuda.set_device(dist.get_rank())
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    if "qwen3" in args.model_name_or_path.lower():
        instruction_template = instruction_template_qwen3
    elif "embeddinggemma" in args.model_name_or_path.lower():
        instruction_template = instruction_template_embeddinggemma

    bench_dict = {
        "mteb_multilingual_v2": "MTEB(Multilingual, v2)",
        "mteb_eng_v2": "MTEB(eng, v2)",
    }

    if args.benchmark:
        benchmark = mteb.get_benchmark(bench_dict[args.benchmark])
        tasks = []
        for task in benchmark.tasks:
            tasks.append(task)
            # if task.metadata.type == "Retrieval":
            #     tasks.append(task)
    else:
        tasks = [mteb.get_task(args.task_name)]

    retrieval_evaluator = evaluate_retrieval(
        tasks=tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        new_inference_mode=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    model = DDP(model, device_ids=[LOCAL_RANK])
    results = retrieval_evaluator.evaluate(model, batch_size=32)

    if rank == 0:
        print(results)

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
