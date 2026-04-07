from inference.test_retrieval_ddp_update import evaluate_retrieval
import os
# for multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    instruction_template_noop,
)
from tasks import EVAL_TASK_DICT
from models.modules import (
    add_pooling_layers,
    last_token_pool,
    mean_pool,
    load_st_dense_layers,
)
from models.t5gemma2model import get_model_t5gemma2_model, load_t5gemma2_encoder
from models.t5gemma2_decoder import get_model_t5gemma2_decoder, load_t5gemma2_decoder
import mteb
from datetime import timedelta
from inference.target_layers import (
    get_qwen_target_layers,
    get_embeddinggemma_target_layers,
)
from inference.helpers import FINAL_EMBEDDING_KEY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, nargs="+", default=None)
    parser.add_argument("--benchmark", type=str, default="mteb_eng_v2")
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="results/performace_evals")
    parser.add_argument(
        "--max_eval_queries",
        type=int,
        default=200000,
        help="Cap the number of queries evaluated per task (random subsample). "
        "Useful for large reranking tasks like MindSmallReranking (~2.4M queries).",
    )
    parser.add_argument("--evaluate_hidden_states", action="store_true")
    parser.add_argument(
        "--t5gemma2_decoder",
        action="store_true",
        help="When evaluating a base T5Gemma2 checkpoint, load the decoder "
        "half instead of the encoder (the same checkpoint contains both).",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", LOCAL_RANK),
        timeout=timedelta(minutes=15),
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
    is_base_model = False
    path_lower = args.model_name_or_path.lower()

    if "qwen3-embedding" in path_lower:
        model_name = "qwen3_embedding"
        if "8b" in path_lower:
            model_name += "_8b"
        elif "0.6b" in path_lower:
            model_name += "_0.6b"
        instruction_template = instruction_template_qwen3
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.pad_token_id

        target_layers_fn = get_qwen_target_layers
    elif "embeddinggemma" in path_lower:
        model_name = "embeddinggemma"
        instruction_template = instruction_template_embeddinggemma
        pool_fn = mean_pool
        add_special_tokens = True
        eot_id = None
        target_layers_fn = get_embeddinggemma_target_layers
        
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
            "t5gemma2_decoder" in path_lower or "t5gemma-2-decoder" in path_lower
        )
        is_t5gemma2 = not is_t5gemma2_decoder and (
            "t5gemma-2" in path_lower or "t5gemma2" in path_lower
        )
        if is_t5gemma2 or is_t5gemma2_decoder:
            pool_fn = mean_pool
            add_special_tokens = True
            eot_id = None
        else:
            pool_fn = last_token_pool
            add_special_tokens = False
            eot_id = tokenizer.eos_token_id
        target_layers_fn = get_qwen_target_layers

    # ---- base (pretrained) models ----
    elif "t5gemma-2" in path_lower or "t5gemma2" in path_lower:
        # Base T5Gemma2 encoder or decoder (not fine-tuned).
        # The same checkpoint (e.g. google/t5gemma-2-270m-270m) contains both
        # encoder and decoder weights; use --t5gemma2_decoder to select the decoder.
        is_t5gemma2_decoder = args.t5gemma2_decoder
        is_t5gemma2 = not is_t5gemma2_decoder
        is_base_model = True
        size_tag = ""
        for tag in ("270m", "540m", "1b", "2b", "4b"):
            if tag in path_lower:
                size_tag = f"_{tag}"
                break
        if is_t5gemma2_decoder:
            model_name = f"t5gemma2_decoder_base{size_tag}"
            pool_fn = last_token_pool
            add_special_tokens = False
            eot_id = tokenizer.eos_token_id
        else:
            model_name = f"t5gemma2_encoder_base{size_tag}"
            pool_fn = mean_pool
            add_special_tokens = True
            eot_id = None
        instruction_template = instruction_template_noop
        target_layers_fn = get_qwen_target_layers

    elif "qwen3" in path_lower:
        # Qwen3 base pretrained (e.g. Qwen3-0.6B)
        model_name = "qwen3_base"
        if "0.6b" in path_lower:
            model_name += "_0.6b"
        elif "1.7b" in path_lower:
            model_name += "_1.7b"
        elif "4b" in path_lower:
            model_name += "_4b"
        instruction_template = instruction_template_noop
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.eos_token_id
        target_layers_fn = get_qwen_target_layers

    elif "gemma-3" in path_lower or "gemma3" in path_lower:
        # Gemma3 base pretrained (causal decoder, e.g. google/gemma-3-1b-pt)
        model_name = "gemma3_base"
        for tag in ("1b", "4b", "12b", "27b"):
            if tag in path_lower:
                model_name += f"_{tag}"
                break
        instruction_template = instruction_template_noop
        pool_fn = last_token_pool
        add_special_tokens = False
        eot_id = tokenizer.eos_token_id
        target_layers_fn = get_qwen_target_layers

    else:
        raise ValueError(
            f"Unrecognized model '{args.model_name_or_path}'. "
            "Expected a path containing 'qwen3', 'embeddinggemma', 'f2llm', "
            "'t5gemma-2', or 'gemma-3'."
        )

    bench_dict = {
        "mteb_multilingual_v2": "MTEB(Multilingual, v2)",
        "mteb_eng_v2": "MTEB(eng, v2)",
        "mteb_eng_v2_subset": "MTEB(eng, v2)",
        "mteb_eng_v2_mini": "MTEB(eng, v2)",
    }

    if args.task_name is not None:
        tasks = mteb.get_tasks(tasks=args.task_name)
    else:
        benchmark = mteb.get_benchmark(bench_dict[args.benchmark])
        tasks = []
        if args.benchmark == "mteb_eng_v2_subset":
            for task in benchmark.tasks:
                if task.metadata.name in EVAL_TASK_DICT["mteb_eng_v2_reduced"]:
                    tasks.append(task)
        elif args.benchmark == "mteb_eng_v2_mini":
            for task in benchmark.tasks:
                if task.metadata.name in EVAL_TASK_DICT["mteb_eng_v2_mini"]:
                    tasks.append(task)
        elif args.benchmark == "mteb_eng_v2":
            for task in benchmark.tasks:
                tasks.append(task)
        
    retrieval_evaluator = evaluate_retrieval(
        tasks=tasks,
        tokenizer=tokenizer,
        instruction_template=instruction_template,
        padding_side="right",
        add_special_tokens=add_special_tokens,
        eot_id=eot_id,
        max_eval_queries=args.max_eval_queries,
    )

    if is_t5gemma2_decoder and is_base_model:
        # Base T5Gemma2 decoder: raw decoder + pooling, no learned projection
        raw_decoder = load_t5gemma2_decoder(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        model = add_pooling_layers(raw_decoder, pool_fn=pool_fn).to("cuda")
    elif is_t5gemma2 and is_base_model:
        # Base T5Gemma2 encoder: raw encoder + pooling, no learned projection
        raw_encoder = load_t5gemma2_encoder(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        model = add_pooling_layers(raw_encoder, pool_fn=pool_fn).to("cuda")
    elif is_t5gemma2_decoder:
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


    #model = DDP(model, device_ids=[LOCAL_RANK])
    #model = torch.compile(model)

    if args.evaluate_hidden_states:
        #model = torch.compile(model)
        m = model.module if hasattr(model, "module") else model
        cfg = m.config
        n_layer = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
        target_layers = list(target_layers_fn(model, n_layer, option="res2", every=2).values())
        target_layers.append(FINAL_EMBEDDING_KEY)
        use_last_token = getattr(m, "pool_fn", None) is last_token_pool
        
        if rank ==0:
            print(target_layers)

        results, summary = retrieval_evaluator.evaluate_hidden_states(
            model,
            batch_size=args.batch_size,
            target_layers=target_layers,
            use_last_token=use_last_token,
        )

    else:
        model = torch.compile(model)
        results, summary = retrieval_evaluator.evaluate(
            model, batch_size=args.batch_size
        )

    filename = ""
    if args.evaluate_hidden_states:
        filename = "_hidden_states"
    if args.filename:
        filename += f"_{args.filename}"
    if args.max_eval_queries:
        filename += f"_{args.max_eval_queries//1000}k_max_queries"

    if rank == 0:
        label = args.benchmark if args.benchmark else "_".join(args.task_name)

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
