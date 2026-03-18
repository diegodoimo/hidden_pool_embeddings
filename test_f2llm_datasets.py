from datasets import load_dataset, Dataset
from tasks.load_datasets import load_task_data
from tasks import get_task

path = "/home/diego/Documents/area_science/ricerca/open/hidden_pool_embeddings/results/f2llm_data_no_instruct/arguana.parquet"
data = load_dataset("parquet", data_files=path)

task = get_task("arguana")
data_minbe = load_task_data("arguana")
