from datasets import load_dataset, Dataset
from tasks.load_datasets import load_task_data
from tasks.helpers import get_task
import torch.distributed as dist

import os

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["LOCAL_RANK"] = "0"

dist.init_process_group()
path = "/home/diego/Documents/area_science/ricerca/open/hidden_pool_embeddings/results/f2llm_annotated/arguana.parquet"
data = load_dataset("parquet", data_files=path)

task = get_task("arguana")
data_mine, *_ = load_task_data(task)

len(sorted(data["train"]["query_text"]))

len(sorted(data_mine["unique_queries"]["text"]))

data["train"]["positive_id"]
data["train"]["qwen3_600m_false_negatives"][0]
data["train"]["qwen3_600m_hard_negatives"][0]


set(data["train"]["negative_id"][0]).intersection(
    set(data["train"]["qwen3_600m_hard_negatives"][0])
)

positive_ids = set(data["train"]["positive_id"])
for i in range(len(data["train"]["qwen3_600m_hard_negatives"])):
    assert set(data["train"]["qwen3_600m_hard_negatives"][i]).issubset(positive_ids), i
