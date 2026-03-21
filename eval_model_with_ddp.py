from inference.test_retrieval_ddp_update import evaluate_retrieval
import os
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from utils.create_datasets import (
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
)
from tasks import EVAL_TASK_DICT
from models.modules import add_pooling_layers, last_token_pool, mean_pool
import mteb
from datetime import timedelta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default="ArguAna")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--out_dir", type=str, default = "results/performace_evals")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(seconds=60),
    )
    rank = dist.get_rank()
    torch.cuda.set_device(LOCAL_RANK)
    torch.set_float32_matmul_precision("high")
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    if "qwen3" in args.model_name_or_path.lower():
        model_name = "qwen3_embedding"
        instruction_template = instruction_template_qwen3
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id
    elif "embeddinggemma" in args.model_name_or_path.lower():
        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None
    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3' or 'embeddinggemma'."
        )

    bench_dict = {
        "mteb_multilingual_v2": "MTEB(Multilingual, v2)",
        "mteb_eng_v2": "MTEB(eng, v2)",
        "mteb_eng_v2_subset": "MTEB(eng, v2)"
    }


    if args.benchmark:
        benchmark = mteb.get_benchmark(bench_dict[args.benchmark])
        tasks = []
        if args.benchmark == "mteb_eng_v2_subset":
            for task in benchmark.tasks:
                if task.metadata.name in EVAL_TASK_DICT["mteb_eng_v2_reduced"]:
                    tasks.append(task)
        else:
            for task in benchmark.tasks:
                tasks.append(task)
    else:
        tasks = [mteb.get_task(args.task_name)]

    retrieval_evaluator = evaluate_retrieval(
        tasks=tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        add_special_tokens=add_special_tokens,
        eot_id=eot_id,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")
    model = add_pooling_layers(model, pool_fn=pool_fn)

    model = DDP(model, device_ids=[LOCAL_RANK])
    model = torch.compile(model)

    results, summary = retrieval_evaluator.evaluate(model, batch_size=64)

    filename = ""
    if args.filename:
        filename = f"_{args.filename}"
    if rank == 0:
        print(results)
        print(summary)
        label = args.benchmark if args.benchmark else args.task_name

        base = os.path.join(args.out_dir, f"{model_name}_{label}{filename}")
        with open(f"{base}_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
