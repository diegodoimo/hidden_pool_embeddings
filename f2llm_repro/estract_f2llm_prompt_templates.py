import pandas as pd
import tqdm
import os
from collections import defaultdict
import json
import argparse
import mteb

benchmark = mteb.get_benchmark("MTEB(eng, v2)")
tasks = list(benchmark.tasks)
tasks_set = set()
for task in tasks:
    tasks_set.add(task.metadata.type)
tasks_set

# root_dir = "/home/diego/.cache/huggingface/hub/datasets--codefuse-ai--F2LLM/snapshots/c8158be982d16202dda93211c1f7a542159acc3e"

# df = pd.read_parquet(f"{root_dir}/arguana.parquet")

# df["query"][0].rstrip("\nQuery: ")
# df["query"][0]


def remove_question(text):
    if "Query: " not in text:
        raise ValueError(f"'Query: ' not found in: {text}")
    prefix, _, _ = text.rpartition("Query: ")
    return prefix + "Query: "


prompts = defaultdict(set)


def main(args):
    for ds_name in tqdm(
        sorted(f for f in os.listdir(args.root_dir) if f.endswith(".parquet"))
    ):
        print(ds_name, flush=True)

        df = pd.read_parquet(f"{args.root_dir}/{ds_name}")
        prompts[ds_name] = set(df["query"].map(remove_question))

    with open("f2llm_prompts.json", "w") as f:
        json.dump({k: list(v) for k, v in prompts.items()}, f, indent=4)


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

        return args

    args = parse_args()

    main(args)
