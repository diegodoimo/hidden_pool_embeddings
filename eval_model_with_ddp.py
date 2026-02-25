from inference.test_retrieval_ddp_update import evaluate_retrieval
import os
import json
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
from inference.helpers import last_token_pool, mean_pool
import mteb
from datetime import timedelta

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
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(
            seconds=60
        ),  
    )
    rank = dist.get_rank()
    torch.cuda.set_device(LOCAL_RANK)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    if "qwen3" in args.model_name_or_path.lower():
        model_name = "qwen3_embedding"
        instruction_template = instruction_template_qwen3
        pool_fn = last_token_pool
        add_special_tokens = False
        append_eos = True
    elif "embeddinggemma" in args.model_name_or_path.lower():
        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        append_eos = False
    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3' or 'embeddinggemma'."
        )

    bench_dict = {
        "mteb_multilingual_v2": "MTEB(Multilingual, v2)",
        "mteb_eng_v2": "MTEB(eng, v2)",
    }

    if args.benchmark:
        benchmark = mteb.get_benchmark(bench_dict[args.benchmark])
        tasks = []
        for task in benchmark.tasks:
            tasks.append(task)
    else:
        tasks = [mteb.get_task(args.task_name)]

    retrieval_evaluator = evaluate_retrieval(
        tasks=tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        new_inference_mode=True,
        pool_fn=pool_fn,
        add_special_tokens=add_special_tokens,
        append_eos=append_eos,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")

    model = DDP(model, device_ids=[LOCAL_RANK])
    model = torch.compile(model)

    results, summary = retrieval_evaluator.evaluate(model, batch_size=64)

    if rank == 0:
        print(results)
        print(summary)
        label = args.benchmark if args.benchmark else args.task_name
        with open(f'{model_name}_{label}_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        with open(f'{model_name}_{label}_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
