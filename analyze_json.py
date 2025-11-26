import json
import numpy as np


path = "/home/diego/Documents/area_science/ricerca/open/hidden_pool_embeddings/results.json"
with open(path, "r") as file:
    data = json.load(file)

len(data)


print(np.mean([instance["main_score"] for instance in data.values()]))
