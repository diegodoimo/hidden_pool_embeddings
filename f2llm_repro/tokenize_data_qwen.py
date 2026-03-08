from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import argparse


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
max_seq_length = 1023


def process_sent(sentence):

    # We make sure there's always an eos token at the end of each sequence
    tokenizer_outputs = tokenizer(
        sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )

    return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])


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
            help="Root directory for tokenized output (e.g. datasets_tokenized)",
        )
        args = parser.parse_args()

        return args

    args = parse_args()

    main(args)
