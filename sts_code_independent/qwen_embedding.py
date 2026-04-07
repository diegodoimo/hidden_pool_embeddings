import os
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from torch import nn

# from utils.create_datasets import TASK_DICT

# from models.modules import (
#     add_pooling_layers,
#     last_token_pool,
#     mean_pool,
#     instruction_template_qwen3,
# )
from datetime import timedelta

import torch
import numpy as np
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)


from embedding_utils import (
    prepare_text_dataset,
    encode_dataset,
    make_collate_fn,
    EvalContext,
    mean_pool,
    add_pooling_layers,
    instruction_template_qwen3,
    instruction_template_embeddinggemma,
    last_token_pool,
)
from datasets import load_dataset


@torch.inference_mode()
def evaluate_one_sts(texts_ds, model, batch_size, eval_context: EvalContext):
    model.eval()

    collate_fn = make_collate_fn(
        eval_context.tokenizer,
        eval_context.padding_side,
        eval_context.eot_id,
        eval_context.add_special_tokens,
    )
    embeddings = encode_dataset(model, texts_ds, batch_size, collate_fn)
    return embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default="ArguAna")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument(
        "--out_file",
        type=str,
        default="embeddings.pt",
        help="Path to save the output .pt file containing 'embeddings' and 'indices'.",
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

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to("cuda")
    model = add_pooling_layers(model, pool_fn=pool_fn)

    model = DDP(model, device_ids=[LOCAL_RANK])
    model = torch.compile(model)

    eval_context = EvalContext(
        tokenizer=tokenizer,
        padding_side="right",
        eot_id=eot_id,
        add_special_tokens=add_special_tokens,
        world_size=WORLD_SIZE,
        rank=RANK,
    )

    dataset = load_dataset("l11p/COCOQA-restval-ICQA")

    all_sentences = [cap[0] for cap in dataset["train"]["captions"]]
    unique_texts, text_to_idx = [], {}
    for text in all_sentences:
        h = hash(text)
        if h not in text_to_idx:
            text_to_idx[h] = len(unique_texts)
            unique_texts.append(text)

    indices1 = [text_to_idx[hash(s)] for s in all_sentences]
    n_dedup = len(all_sentences) - len(unique_texts)

    task_metadata = ""
    texts_ds = prepare_text_dataset(
        unique_texts, task_metadata, instruction_template, tokenizer, rank
    )
    embeddings = evaluate_one_sts(texts_ds, model, 32, eval_context)
    # Keep on CPU as float32; shape: [n_unique, embed_dim]
    embeddings_cpu = embeddings.cpu()
    # LongTensor of shape [n_all]: maps each original caption to its row in embeddings_cpu
    indices_tensor = torch.tensor(indices1, dtype=torch.long)

    if rank == 0:
        print(f"Encoded {len(unique_texts)} unique texts ({n_dedup} duplicates removed)")
        print(f"embeddings shape : {embeddings_cpu.shape}")
        print(f"indices shape    : {indices_tensor.shape}")

        out_path = args.out_file
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings_cpu,   # [n_unique, embed_dim]
                "indices": indices_tensor,       # [n_all]  caption_i -> embeddings[indices[i]]
            },
            out_path,
        )
        print(f"Saved embeddings and indices to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    RANK = int(os.environ["RANK"])
    main()
