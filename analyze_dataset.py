import json

with open("results/dataset_stats.json", "r") as f:
    data = json.load(f)


sorted_keys = sorted(
    data.keys(),
    key=lambda k: data[k]["unique_queries"] + data[k]["unique_documents"],
    reverse=True,
)

total_unique_queries = sum(data[k]["unique_queries"] for k in data)
total_unique_docs = sum(data[k]["unique_documents"] for k in data)

col1_w = max(len(k) for k in data.keys())
col2_w = max(len(str(data[k]["unique_queries"])) for k in data)
col3_w = max(len(str(data[k]["unique_documents"])) for k in data)

with open("dataset_list.txt", "w") as f:
    header = f"{'dataset':<{col1_w}}  {'unique_queries':>{col2_w}}  {'unique_documents':>{col3_w}}"
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for key in sorted_keys:
        num_unique_q = data[key]["unique_queries"]
        num_unique_doc = data[key]["unique_documents"]
        f.write(
            f"{key:<{col1_w}}  {num_unique_q:>{col2_w}}  {num_unique_doc:>{col3_w}}\n"
        )
    f.write("-" * len(header) + "\n")
    f.write(
        f"{'TOTAL':<{col1_w}}  {total_unique_queries:>{col2_w}}  {total_unique_docs:>{col3_w}}\n"
    )
