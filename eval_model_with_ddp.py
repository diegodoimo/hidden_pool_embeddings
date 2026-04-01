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
    instruction_template_f2llm,
)
from tasks import EVAL_TASK_DICT
from models.modules import add_pooling_layers, last_token_pool, mean_pool, load_st_dense_layers
from models.t5gemma2model import get_model_t5gemma2_model
from models.t5gemma2_decoder import get_model_t5gemma2_decoder
import mteb
from datetime import timedelta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default="ArguAna")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default = "results/performace_evals")
    parser.add_argument(
        "--max_eval_queries",
        type=int,
        default=None,
        help="Cap the number of queries evaluated per task (random subsample). "
        "Useful for large reranking tasks like MindSmallReranking (~2.4M queries).",
    )
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

    is_t5gemma2 = False
    is_t5gemma2_decoder = False
    path_lower = args.model_name_or_path.lower()

    if "qwen3" in path_lower and "f2llm" not in path_lower:
        model_name = "qwen3_embedding"
        if "8b" in path_lower:
            model_name += "_8b"
        elif "0.6b" in path_lower:
            model_name += "_0.6b"
        instruction_template = instruction_template_qwen3
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id
    elif "embeddinggemma" in path_lower:
        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None
    elif "f2llm" in path_lower:
        model_name = "f2llm"
        if "0.6b" in path_lower:
            model_name += "_0.6b"
        elif "1.7b" in path_lower:
            model_name += "_1.7b"
        elif "4b" in path_lower:
            model_name += "_4b"
        instruction_template = instruction_template_f2llm
        is_t5gemma2_decoder = (
            "t5gemma2_decoder" in path_lower
            or "t5gemma-2-decoder" in path_lower
        )
        is_t5gemma2 = (
            not is_t5gemma2_decoder
            and ("t5gemma-2" in path_lower or "t5gemma2" in path_lower)
        )
        if is_t5gemma2 or is_t5gemma2_decoder:
            pool_fn = mean_pool
            add_special_tokens = True
            eot_id = None
        else:
            pool_fn = last_token_pool
            add_special_tokens = False
            eot_id = tokenizer.eos_token_id
    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3', 'embeddinggemma', or 'f2llm'."
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
        max_eval_queries=args.max_eval_queries,
    )

    if is_t5gemma2_decoder:
        model, _, _ = get_model_t5gemma2_decoder(
            model_name_or_path=args.model_name_or_path,
            activation_checkpointing=False,
            attention_pooling=False,
            attention_dim=None,
        )
        model = model.to("cuda")
    elif is_t5gemma2:
        model, _, _ = get_model_t5gemma2_model(
            model_name_or_path=args.model_name_or_path,
            activation_checkpointing=False,
            attention_pooling=False,
            cls_query_pooling=False,
            attention_dim=None,
        )
        model = model.to("cuda")
    else:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            dtype=torch.bfloat16,
        ).to("cuda")
        projection = load_st_dense_layers(args.model_name_or_path, dtype=torch.bfloat16)
        if projection is not None:
            projection = projection.to("cuda")
        model = add_pooling_layers(model, pool_fn=pool_fn, projection_layers=projection)

    model = DDP(model, device_ids=[LOCAL_RANK])
    model = torch.compile(model)

    results, summary = retrieval_evaluator.evaluate(model, batch_size=args.batch_size)

    filename = ""
    if args.filename:
        filename = f"_{args.filename}"
    if args.max_eval_queries:
        filename+=f"_{args.max_eval_queries//1000}k_max_queries"
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
