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

from inference.test_retrieval_ddp_update import compute_averages
import time
from collections import defaultdict
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default="ArguAna")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
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
    world_size = dist.get_world_size() 

    if "qwen3" in args.model_name_or_path.lower():
        model_name = "qwen3_embedding"
        if "8b" in args.model_name_or_path.lower():
            model_name+="_8b"
        elif "0.6b" in args.model_name_or_path.lower():
            model_name += "_0.6b"
    elif "embeddinggemma" in args.model_name_or_path.lower():
        model_name = "embeddinggemma"
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


    print("loading model")
    model = mteb.get_model(args.model_name_or_path)

    print("evaluating tasks")
    start0 = time.time()
    results = {}
    results_by_type = defaultdict(list)
    for i, task in enumerate(tasks):

        # double check that this aling with the one shot MTEB evaluate benchmark 
        print(
            f"evaluating task: {task.metadata.name} ({i+1}/{len(tasks)}) {(time.time()-start0)/60:.1f}min"
        )
        start = time.time()
        res = mteb.evaluate(model, tasks=task, overwrite_strategy="always", encode_kwargs={"batch_size": args.batch_size})
        end = time.time()
        duration = end - start
        instance = res.task_results[0]
        splits = set(instance.scores.keys())
        if len(splits) > 1:
            print(f"{task.metadata.name} has more than 1 split {splits}")
        split = "test" if "test" in splits else next(iter(splits))
        main_score = instance.scores[split][0]["main_score"]
        results[instance.task_name] = {
            "main_score": main_score,
            "task_type": instance.task_type,
            "time": f"{duration:.2f}",
            "split": split,
        }

        print(results)
        print(np.mean([score["main_score"] for score in results.values()]))
        results_by_type[instance.task_type].append({instance.task_name: ({"main_score": main_score}, duration)})

    summary = compute_averages(results_by_type)

    filename = ""
    if args.filename:
        filename = f"_{args.filename}"

    print(results)
    print(summary)
    label = args.benchmark if args.benchmark else args.task_name

    base = os.path.join(args.out_dir, f"mteb_{model_name}_{label}{filename}")
    with open(f"{base}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    assert WORLD_SIZE==1
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
