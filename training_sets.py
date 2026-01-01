import concurrent.futures
import requests
import mteb


kalm_eng_trainingset = [
    "CodeFeedback",
    "ELI5",
    "ExpertQA",
    "GooAQ",
    "MEDI2BGE",
    "OpenOrca",
    "PAQ",
    "PubMedQA",
    "SearchQA",
    "arxiv_qa",
    "CC-News",
    "TREC-COVID",
    "DBpedia-Entity",
    "ESCI",
    "FEVER",
    "FiQA",
    "HotpotQA",
    "MLDR",
    "MSMARCO",
    "MSMARCO-v2",
    "NFCorpus",
    "rag-dataset-12000",
    "SciFact",
    "SQuAD 2.0",
    "TriviaQA",
    "WebGPT Comparisons",
    "Natural Questions",
    "Yahoo Answers",
    "CQADupStack",
    "ContractNLI",
    "MultiNLI",
    "NLLB",
    "Quora",
    "WikiAnswers",
    "SimCSE NLI",
    "SNLI",
    "arXiv Classfication",
    "Biorxiv Classfication",
    "Medrxiv Classfication",
    "Reddit-Clustering",
    "Reddit-Clustering-P2P",
    "Stackexchange-Clustering",
    "Stackexchange-Clustering-P2P",
    "TwentyNewsgroups-Clustering",
    "AmazonPolarity",
    "IMDB",
    "banking77",
    "EmotionClassification",
    "TweetSentimentExtraction",
    "ToxicConversations",
]


API_URL = "https://datasets-server.huggingface.co/splits"
DATASET_INFO_URL = "https://datasets-server.huggingface.co/info"
SIZE_URL = "https://datasets-server.huggingface.co/size"


tasks = mteb.get_tasks(modalities=["text"], exclusive_modality_filter=True)
results = {}
MAX_WORKERS = 20  # Adjust this based on your network bandwidth and server limits


# The function that performs the processing for a single task
def process_task(task):
    name = task.metadata.dataset["path"]
    if name.lower() == "aggregate tasks do not have a path":
        return None, None  # Return None to indicate no result

    eval_splits = task.metadata.eval_splits

    # 1. Get available splits
    try:
        response = requests.get(API_URL, params={"dataset": name}, timeout=10)
        response.raise_for_status()
        splits_info = response.json().get("splits", [])
        split_names = set(item["split"].lower() for item in splits_info)

        # Remove eval splits from available splits
        for eval_split in eval_splits:
            split_names.discard(eval_split)  # discard won't raise error if item not found

        if not split_names:
            return name, None  # No training splits found

        # 2. Get dataset info (including language)
        info_response = requests.get(DATASET_INFO_URL, params={"dataset": name}, timeout=10)
        info_response.raise_for_status()
        info_data = info_response.json()

        languages = []
        if "dataset_info" in info_data:
            for config_data in info_data["dataset_info"].values():
                if isinstance(config_data, dict) and "languages" in config_data:
                    languages = config_data.get("languages", [])
                    break

        # 3. Get size information
        size_response = requests.get(SIZE_URL, params={"dataset": name}, timeout=10)
        size_response.raise_for_status()
        size_data = size_response.json()

        train_split_info = {}
        if "size" in size_data and "splits" in size_data["size"]:
            for split in size_data["size"]["splits"]:
                split_name = split["split"].lower()
                if split_name in split_names:
                    num_rows = split.get("num_rows", "unknown")

                    train_split_info[split_name] = {
                        "num_rows": split.get("num_rows", "unknown"),
                    }

        if train_split_info:
            result = {
                "test_split": eval_splits,
                "train_splits": list(split_names),
                "languages": languages if languages else "unknown",
                "train_split_info": train_split_info,
            }
        else:
            result = None
        return name, result

    except requests.exceptions.RequestException as e:
        print(f"Error processing dataset {name}: {e}")
        return name, None
    except Exception as e:
        print(f"An unexpected error occurred for dataset {name}: {e}")
        return name, None


# Use ThreadPoolExecutor to run tasks concurrently
MAX_TASKS = 100
tasks_to_run = tasks
count = 0
total_tasks = len(tasks_to_run)


with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # map submits the tasks and returns an iterator of results
    future_to_task = {executor.submit(process_task, task): i for i, task in enumerate(tasks_to_run)}

    for future in concurrent.futures.as_completed(future_to_task):
        task_index = future_to_task[future]
        count += 1

        try:
            name, result = future.result()
            if result is not None and name is not None:
                results[name] = result
        except Exception as exc:
            print(f"Task {task_index} generated an exception: {exc}")

        # Simple progress tracking
        if count % 100 == 0 or count == total_tasks:
            print(f"{count}/{total_tasks} tasks evaluated. Found {len(results)} results.")


train_set = {}
for key, val in results.items():

    splits = val["train_splits"]
    split = "train" if "train" in splits else splits[0]
    data_size = val["train_split_info"][split]["num_rows"]
    train_set[key] = data_size

val["train_splits"]

my_set = set(elem.split("/")[1].lower() for elem in list(train_set.keys()))

kalm_lower = set(elem.lower() for elem in kalm_eng_trainingset)
my_set.intersection(kalm_lower)


sum(list(train_set.values())) / 10**9
sorted_items = sorted(train_set.items(), key=lambda item: -item[1])
sorted_items

SIZE_URL = "https://datasets-server.huggingface.co/size"
name = "mteb/msmarco-v2"
size_response = requests.get(SIZE_URL, params={"dataset": name}, timeout=10)
size_response.raise_for_status()
size_data = size_response.json()


task = mteb.get_task("mteb/msmarco-v2"])


name = task.metadata.dataset["path"]
if name.lower() == "aggregate tasks do not have a path":
    return None, None  # Return None to indicate no result

eval_splits = task.metadata.eval_splits



response = requests.get(API_URL, params={"dataset": name}, timeout=10)
response.raise_for_status()
splits_info = response.json().get("splits", [])
split_names = set(item["split"].lower() for item in splits_info)

# Remove eval splits from available splits
for eval_split in eval_splits:
    split_names.discard(eval_split)  # discard won't raise error if item not found

if not split_names:
    return name, None  # No training splits found

# 2. Get dataset info (including language)
info_response = requests.get(DATASET_INFO_URL, params={"dataset": name}, timeout=10)
info_response.raise_for_status()
info_data = info_response.json()

languages = []
if "dataset_info" in info_data:
    for config_data in info_data["dataset_info"].values():
        if isinstance(config_data, dict) and "languages" in config_data:
            languages = config_data.get("languages", [])
            break

# 3. Get size information
size_response = requests.get(SIZE_URL, params={"dataset": name}, timeout=10)
size_response.raise_for_status()
size_data = size_response.json()

train_split_info = {}
if "size" in size_data and "splits" in size_data["size"]:
    for split in size_data["size"]["splits"]:
        split_name = split["split"].lower()
        if split_name in split_names:
            num_rows = split.get("num_rows", "unknown")

            train_split_info[split_name] = {
                "num_rows": split.get("num_rows", "unknown"),
            }
