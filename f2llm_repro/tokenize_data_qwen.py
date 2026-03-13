from multiprocessing import Pool
import numpy as np
import pandas as pd
import os

from transformers import AutoTokenizer
from tqdm.auto import tqdm
import argparse


# ---- globals set by main() before any worker fork ----
tokenizer = None
max_seq_length = None
_add_special_tokens = None
_append_eos = None


# Model-type presets:
#   qwen3      – add_special_tokens=False, manually append eos_token_id,
#                max_seq_length reserves 1 slot for that EOS.
#   t5-gemma2  – add_special_tokens=True (tokenizer handles BOS/EOS),
#                no manual EOS append.
MODEL_TYPE_PRESETS = {
    "qwen3": {"add_special_tokens": False, "append_eos": True, "seq_len_reserve": 1},
    "t5-gemma2": {
        "add_special_tokens": True,
        "append_eos": False,
        "seq_len_reserve": 0,
    },
}


def _infer_model_type(tokenizer_path: str) -> str:
    """Infer model type from the tokenizer path string."""
    path_lower = tokenizer_path.lower()
    if "t5gemma" in path_lower or "t5-gemma" in path_lower:
        return "t5-gemma2"
    if "qwen" in path_lower:
        return "qwen3"
    raise ValueError(
        f"Cannot infer model type from tokenizer_path '{tokenizer_path}'. "
        f"Expected the path to contain 'qwen' or 't5gemma'. "
        f"Known model types: {list(MODEL_TYPE_PRESETS.keys())}"
    )


def process_sent(sentence):

    tokenizer_outputs = tokenizer(
        sentence,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=_add_special_tokens,
    )

    ids = tokenizer_outputs.input_ids
    if _append_eos:
        ids = ids + [tokenizer.eos_token_id]

    return np.array(ids)


def process_sent_batch(s):
    return s.apply(process_sent)


def parallelize(data, func, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data


def main(args):

    for ds_name in tqdm(
        sorted(f for f in os.listdir(args.root_dir) if f.endswith(".parquet"))
    ):
        print(ds_name, flush=True)

        df = pd.read_parquet(f"{args.root_dir}/{ds_name}")
        df["query_input_ids"] = parallelize(
            df["query"], process_sent_batch, args.num_workers
        )

        num_neg = 24 if "negative_2" in df.keys() else 1

        ls = df.passage.to_list()
        for i in range(1, num_neg + 1):
            ls += df[f"negative_{i}"].to_list()
        ls = list(set(ls))
        df_tmp = pd.DataFrame({"text": ls})
        df_tmp["input_ids"] = parallelize(
            df_tmp["text"], process_sent_batch, args.num_workers
        )
        df_tmp = df_tmp.set_index("text")

        df["passage_input_ids"] = df.passage.map(df_tmp.input_ids)

        for i in range(1, num_neg + 1):
            df[f"negative_{i}_input_ids"] = df[f"negative_{i}"].map(df_tmp.input_ids)

        os.makedirs(args.output_dir, exist_ok=True)
        df.to_parquet(f"{args.output_dir}/{ds_name}", index=False)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Tokenize datasets from datasets_negatives into datasets_tokenized."
        )
        parser.add_argument(
            "--root_dir",
            type=str,
            help="Root directory containing hard-negative datasets (e.g. datasets_negatives)",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Root directory for tokenized output (e.g. datasets_tokenized)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            help="Number of parallel workers for tokenization",
        )
        parser.add_argument(
            "--tokenizer_path",
            type=str,
            required=True,
            help="HuggingFace model path for the tokenizer (e.g. Qwen/Qwen3-0.6B or google/t5gemma-2-270m-270m)",
        )
        parser.add_argument(
            "--max_seq_len",
            type=int,
            default=1024,
            help="Maximum total sequence length including any special tokens (default: 1024)",
        )
        args = parser.parse_args()

        return args

    args = parse_args()

    # ---- configure globals before forking workers ----
    model_type = _infer_model_type(args.tokenizer_path)
    preset = MODEL_TYPE_PRESETS[model_type]
    _add_special_tokens = preset["add_special_tokens"]
    _append_eos = preset["append_eos"]
    max_seq_length = args.max_seq_len - preset["seq_len_reserve"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    print(
        f"Tokenizer : {args.tokenizer_path}\n"
        f"Model type: {model_type}\n"
        f"add_special_tokens={_add_special_tokens}, append_eos={_append_eos}, "
        f"max_seq_length={max_seq_length} (total cap {args.max_seq_len})"
    )

    main(args)
