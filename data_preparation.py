from datasets import load_dataset
import argparse
from sentence_transformers import SentenceTransformer
import sys
import torch
import mteb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()
    return args


# def main():

sys.argv = [""]
args = parse_args()


args.model_name_or_path = "google/embeddinggemma-300m"
model = SentenceTransformer(
    args.model_name_or_path,
    trust_remote_code=True,
    model_kwargs={
        "dtype": "auto",
    },
).to(device="cuda")

hf_corpus = load_dataset("mteb/msmarco", name="corpus")
hf_queries = load_dataset("mteb/msmarco", name="queries")
hf_qrels = load_dataset("mteb/msmarco", name="default")


hf_corpus
hf_queries
hf_qrels


task = mteb.get_tasks(["MSMARCO"])[0]


task.load_data()

dataset = load_dataset("mteb/msmarco")
dataset = load_dataset("mteb/msmarco")
dataset

dataset["train"]


if __name__ == "__main__":
    main()
