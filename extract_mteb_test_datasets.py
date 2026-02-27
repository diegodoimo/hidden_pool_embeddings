import argparse
import mteb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract datasets and test splits from an MTEB benchmark."
    )
    # parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark (e.g., mteb_eng_v2, MTEB(eng, v2))")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/mteb_datasets",
        help="Path to the output text file. If not provided, prints to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Mapping for common short names to full MTEB benchmark names
    bench_dict = {
        "mteb_multilingual_v2": "MTEB(Multilingual, v2)",
        "mteb_eng_v2": "MTEB(eng, v2)",
    }
    for filename, benchmark_name in bench_dict.items():
        # benchmark_name = bench_dict.get(args.benchmark, args.benchmark)

        print(f"Loading benchmark: {filename}...")
        benchmark = mteb.get_benchmark(benchmark_name)

        with open(f"{args.output_file}_benchmark_{filename}", "w") as f:
            f.write(f"{'Dataset Name':<50} | {'Test Split':<15}")
            f.write("-" * 70)

        for task in benchmark.tasks:
            dataset_name = task.metadata.name

            # Determine the split used for testing
            eval_splits = []
            if hasattr(task, "metadata") and hasattr(task.metadata, "eval_splits"):
                eval_splits = task.metadata.eval_splits
            elif hasattr(task, "eval_splits"):
                eval_splits = task.eval_splits

            test_split = "unknown"
            if eval_splits:
                if "test" in eval_splits:
                    test_split = "test"
                elif "validation" in eval_splits:
                    test_split = "validation"
                elif "dev" in eval_splits:
                    test_split = "dev"
                else:
                    test_split = eval_splits[0]

            with open(f"{args.output_file}_{filename}", "a") as f:
                f.write(f"{dataset_name:<50} | {test_split:<15}\n")


if __name__ == "__main__":
    main()
