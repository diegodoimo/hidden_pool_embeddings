import json

with open("results/dataset_stats.json", "r") as f:
    data = json.load(f)


data.keys()

total_unique = sum([val["total_queries"] for val in data.values()]) / 10**6
total_unique


total_unique = sum([val["unique_documents"] for val in data.values()]) / 10**6

with open("dataset_list.txt", "w") as f:
    for key in sorted(data.keys()):
        num_unique_q = data[key]["unique_queries"]
        num_unique_doc = data[key]["unique_documents"]
        f.write(key + "\t" + str(num_unique_q) + "\t" + str(num_unique_doc) + "\n")
