from tasks.abs_task import AbsTask, TaskMetadata
from datasets import load_dataset, concatenate_datasets, Dataset
import time
import torch.distributed as dist
import pandas as pd
import numpy as np
from tasks.data_helpers import RetrievalRawData
from utils.helpers import return_formatted
from tasks.retrieval_tasks.retrieval_loaders import from_one_hf_dataset


# List of all 174 StackExchange subjects
STACKEXCHANGE_SUBJECTS = [
    "3dprinting", "academia", "ai", "android", "anime", "apple", "arduino", "askubuntu",
    "astronomy", "avp", "aviation", "beer", "bicycles", "bioinformatics", "biology",
    "bitcoin", "blender", "boardgames", "bricks", "buddhism", "chemistry", "chess",
    "christianity", "civicrm", "codegolf", "codereview", "coffee", "cogsci", "computergraphics",
    "conlang", "cooking", "craftcms", "crafts", "crypto", "cs", "cseducators", "cstheory",
    "datascience", "dba", "devops", "diy", "drupal", "dsp", "earthscience", "ebooks",
    "economics", "electronics", "elementaryos", "ell", "emacs", "engineering", "english",
    "ethereum", "expatriates", "expressionengine", "fitness", "freelancing", "french",
    "gamedev", "gaming", "gardening", "genealogy", "german", "gis", "graphicdesign",
    "ham", "hardwarerecs", "health", "hermeneutics", "hin", "history", "hobbyists",
    "homebrew", "hsm", "hsm-history", "iot", "islam", "italian", "japanese", "joomla",
    "judaism", "korean", "languagelearning", "latin", "law", "libertarianism", "lifehacks",
    "linguistics", "literature", "magento", "martialarts", "materials", "mathematica",
    "math", "matheducators", "mathoverflow", "mechanics", "money", "monero", "movies",
    "music", "musicfans", "mythology", "networkengineering", "opendata", "opensource",
    "outdoors", "parenting", "patents", "pets", "philosophy", "philosophy-of-language",
    "photo", "physics", "pm", "poker", "politics", "portuguese", "productivity",
    "proofassistants", "psychology", "pt", "puzzling", "quant", "quantumcomputing",
    "raspberrypi", "retrocomputing", "reverseengineering", "robotics", "rpg", "rus",
    "russian", "salesforce", "scicomp", "scifi", "security", "sharepoint", "sitecore",
    "skeptics", "softwareengineering", "solana", "sound", "space", "sports", "sqa",
    "stackapps", "stats", "stellar", "success", "superuser", "sustainability", "tex",
    "tezos", "tor", "travel", "tridion", "unix", "ux", "vi", "webapps", "webmasters",
    "wine", "woodworking", "wordpress", "workplace", "worldbuilding", "writing",
]


def load_stackexchange_all_subjects(task, max_num_queries=10**6, rank=None) -> RetrievalRawData:
    """Load all 174 StackExchange subjects, concatenate them, and process as a single dataset.
    
    This loader:
    1. Loads all 174 StackExchange subject subsets
    2. Concatenates them into a single dataset
    3. Combines title + body into a single query field
    4. Uses the standard from_one_hf_dataset processing logic
    
    Args:
        task: Task object with dataset configuration
        max_num_queries: Maximum number of queries to keep (default: 1 million)
        rank: Distributed training rank (if None, obtained from dist.get_rank())
    """
    rank = dist.get_rank() if rank is None else rank
    
    if rank == 0:
        start_total = time.time()
        print(f"Loading all {len(STACKEXCHANGE_SUBJECTS)} StackExchange subjects...")
    
    # Load and concatenate all subjects
    all_datasets = []
    for i, subject in enumerate(STACKEXCHANGE_SUBJECTS):
        try:
            if rank == 0 and i % 20 == 0:
                print(f"  Loading subject {i+1}/{len(STACKEXCHANGE_SUBJECTS)}: {subject}")
            
            dataset = load_dataset(task.hf_name, name=subject, split=task.split)
            all_datasets.append(dataset)
            
        except Exception as e:
            if rank == 0:
                print(f"  Warning: Failed to load {subject}: {e}")
            continue
    
    dist.barrier()
    if rank == 0:
        print(f"Loaded {len(all_datasets)} subjects in {(time.time()-start_total)/60:.2f} min")
        print("Concatenating datasets...")
        start = time.time()
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(all_datasets)
    
    dist.barrier()
    if rank == 0:
        print(f"Concatenation done in {(time.time()-start)/60:.2f} min")
        print(f"Total dataset size: {return_formatted(len(combined_dataset))}")
        print("Combining title and body into query field...")
        start = time.time()
    
    # Combine title and body into a single query field
    def combine_title_body(example):
        example[task.anchor_name] = example["title"] + " " + example["body"]
        return example
    
    combined_dataset = combined_dataset.map(
        combine_title_body,
        batched=False,
        desc="Combining title+body" if rank == 0 else None,
    )
    
    dist.barrier()
    if rank == 0:
        print(f"Title+body combination done in {(time.time()-start)/60:.2f} min")
        print(f"Processing with unified loader...")
    
    # Create a temporary task-like object for from_one_hf_dataset
    # We'll modify the dataset in place and pass the task object
    # The from_one_hf_dataset expects to load the dataset itself, so we need to
    # work around this by temporarily storing the combined dataset
    
    # Actually, let's just replicate the from_one_hf_dataset logic here with our combined dataset
    # This is cleaner than trying to monkey-patch
    
    start = time.time()
    n_pairs = len(combined_dataset)
    
    if rank == 0:
        print("Converting to pandas...")
    
    # Convert to pandas DataFrame
    has_title = task.corpus_fields.get("title", None) is not None
    cols_to_load = [task.anchor_name, task.positive_name]
    if has_title:
        title_col = task.corpus_fields.get("title", None)
        if title_col in combined_dataset.column_names:
            cols_to_load.append(title_col)
        else:
            has_title = False
    
    df = combined_dataset.select_columns(cols_to_load).to_pandas()
    
    # Keep as pandas Series
    query_texts = df[task.anchor_name]
    positive_texts = df[task.positive_name]
    
    dist.barrier()
    if rank == 0:
        print(f"Conversion done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Finding unique queries and positives...")
    
    # Fast deduplication via pandas
    unique_query_mask = ~query_texts.duplicated(keep="first")
    unique_query_idx = unique_query_mask[unique_query_mask].index
    unique_query_texts = query_texts.iloc[unique_query_idx].reset_index(drop=True)
    unique_query_ids = [f"query_{i}" for i in unique_query_idx]
    
    unique_positive_mask = ~positive_texts.duplicated(keep="first")
    unique_positive_idx = unique_positive_mask[unique_positive_mask].index
    unique_positive_texts = positive_texts.iloc[unique_positive_idx].reset_index(drop=True)
    unique_positive_ids = [f"doc_{i}" for i in unique_positive_idx]
    n_positives = len(unique_positive_ids)
    
    if has_title:
        unique_positive_titles = df[title_col].iloc[unique_positive_idx].reset_index(drop=True)
    else:
        unique_positive_titles = None
    
    dist.barrier()
    if rank == 0:
        print(f"Deduplication done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Remapping indices...")
    
    # Vectorized remapping
    query_text_to_first_idx = pd.Series(
        unique_query_idx.values, index=query_texts.iloc[unique_query_idx].values
    )
    positive_text_to_first_idx = pd.Series(
        unique_positive_idx.values,
        index=positive_texts.iloc[unique_positive_idx].values,
    )
    
    query_ids = ("query_" + query_texts.map(query_text_to_first_idx).astype(str)).tolist()
    positive_ids = ("doc_" + positive_texts.map(positive_text_to_first_idx).astype(str)).tolist()
    
    dist.barrier()
    if rank == 0:
        print(f"Remapping done in {(time.time()-start)/60:.2f} min")
        start = time.time()
        print("Generating corpus dict...")
    
    # Build corpus dict
    if has_title:
        corpus_dict = {
            id_: {"text": doc_text, "title": doc_title}
            for id_, doc_text, doc_title in zip(
                unique_positive_ids, unique_positive_texts, unique_positive_titles
            )
        }
    else:
        corpus_dict = {
            id_: {"text": doc_text}
            for id_, doc_text in zip(unique_positive_ids, unique_positive_texts)
        }
    
    dist.barrier()
    if rank == 0:
        print(f"Corpus dict built in {(time.time()-start)/60:.2f} min")
    
    # Apply query limiting if needed
    if max_num_queries is not None and len(unique_query_idx) > max_num_queries:
        if rank == 0:
            start = time.time()
            print(
                f"Number of unique queries {return_formatted(len(unique_query_idx))} > {max_num_queries//10**6}M: limiting queries"
            )
        
        from tasks.retrieval_tasks.retrieval_loaders import limit_number_of_queries
        
        unique_query_texts = unique_query_texts[:max_num_queries]
        unique_query_ids = unique_query_ids[:max_num_queries]
        unique_query_idx = unique_query_idx[:max_num_queries]
        
        (
            query_ids,
            positive_ids,
            unique_positive_ids,
            unique_positive_texts,
            unique_positive_titles,
            n_positives,
        ) = limit_number_of_queries(
            query_ids=query_ids,
            positive_ids=positive_ids,
            unique_query_idx=unique_query_idx,
            n_pairs=n_pairs,
            unique_positive_ids=unique_positive_ids,
            unique_positive_texts=unique_positive_texts,
            unique_positive_titles=unique_positive_titles,
            has_title=has_title,
            max_queries=max_num_queries,
        )
        
        if rank == 0:
            print(f"Queries limited in {(time.time()-start)/60:.2f} min")
    
    dist.barrier()
    
    assert set(positive_ids).issubset(
        set(unique_positive_ids)
    ), "filtered qrels contain positive IDs not in corpus"
    
    assert set(unique_positive_ids) == set(corpus_dict.keys())
    
    if rank == 0:
        print(f"Found {return_formatted(len(unique_query_texts))} unique queries")
        print(f"Total number of query-positive pairs: {return_formatted(len(query_ids))}")
        print(f"Positives referenced by pairs (n_positives): {return_formatted(n_positives)}")
        print(f"Total unique documents in corpus: {return_formatted(len(unique_positive_ids))}")
        print(f"Total processing time: {(time.time()-start_total)/60:.2f} min")
    
    return RetrievalRawData(
        query_ids=query_ids,
        positive_ids=positive_ids,
        document_texts=unique_positive_texts,
        document_ids=unique_positive_ids,
        document_titles=unique_positive_titles,
        unique_query_texts=unique_query_texts,
        unique_query_ids=unique_query_ids,
        corpus_dict=corpus_dict,
        has_title=has_title,
        n_positives=n_positives,
    )


class StackExchangeRetrieval(AbsTask):
    """StackExchange dataset for retrieval - all 174 subjects concatenated.
    
    Title+body is combined into a single query field, upvoted_answer is used as positive.
    Loads and concatenates all 174 StackExchange subject subsets.
    """

    language = "en"
    hf_name = "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "title_body"  # This field will be created by combining title + body
    positive_name = "upvoted_answer"
    corpus_fields = {}  # No title field for answers, loader will handle has_title check
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve answers that best answer the question"
        },
    )
    loader = load_stackexchange_all_subjects
