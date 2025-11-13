import argparse
from sentence_transformers import SentenceTransformer
import mteb
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--task")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="./tokenizer")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=6)
    parser.add_argument("--avg_token", action="store_true")
    parser.add_argument("--save_distances", action="store_true")
    parser.add_argument("--remove_duplicates", action="store_true")
    parser.add_argument("--target_layer", type=str, default="res2")
    parser.add_argument("--maxk", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--include_last_k", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--samples_per_label", type=int, default=None)
    parser.add_argument("--dataset_reduce_factor", type=int, default=1)
    parser.add_argument("--only_output", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    bench_dict = {"mteb_multilingual_v2": "MTEB(Multilingual, v2)", "mteb_eng_v2": "MTEB(eng, v2)"}
    tasks_dict = {"classification", "retrieval", "clustering"}

    model = mteb.get_model(args.model_name)

    benchmark = mteb.get_benchmark(bench_dict[args.benchmark])
    tasks = []
    for task in benchmark.tasks:
        if task.metadata.type == "Retrieval" and len(tasks) < 5:
            tasks.append(task)

    print("evaluating tasks")
    start = time.time()
    results = mteb.evaluate(model, tasks=tasks)
    end = time.time()
    print(results, f"{(end-start)/60}min")


if __name__ == "__main__":
    main()
