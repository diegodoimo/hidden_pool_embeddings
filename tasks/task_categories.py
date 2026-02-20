"""
Task categorization for organizing all task types.
Maps task names to their respective categories.
"""

# ============================================================================
# RETRIEVAL TASK CATEGORIES
# ============================================================================

OPEN_DOMAIN_QA = [
    "naturalquestions",
    "triviaqa",
    "paq",
    "eli5",
    "squad",
    "hotpotqa",
    "gooaq",
    "yahooanswers",
]

DOMAIN_SPECIFIC_QA = [
    "bioasq",
    "pubmedqa",
    "fiqa2018",
    "amazonqa",
    "coliee",
]

GENERAL_RETRIEVAL = [
    "msmarco",
    "nfcorpus",
    "stackexchange",
    "miracl",
    "mrtydi",
    "arguana",
]

FACT_VERIFICATION = [
    "fever",
    "scifact",
]

PARAPHRASE_DETECTION = [
    "qqp",
    "stackexchange_dup_p2p",
    "stackexchange_dup_s2s",
    "stackoverflow_dup",
]

SCIENTIFIC_DOC_RETRIEVAL = [
    "s2orc_abstract_citation",
    "s2orc_title_abstract",
    "s2orc_title_citation",
    "specter",
]

SUMMARIZATION = [
    "cnndm",
    "xsum",
    "sentence_compression",
    "wikihow",
]

# ============================================================================
# NLI TASK CATEGORIES
# ============================================================================

NLI_TASKS = [
    "snli",
    "mnli",
    "anli",
    "all_nli",  # Combined NLI dataset
]

XNLI_TASKS = [
    "xnli_ar",
    "xnli_bg",
    "xnli_de",
    "xnli_el",
    "xnli_es",
    "xnli_fr",
    "xnli_hi",
    "xnli_ru",
    "xnli_sw",
    "xnli_th",
    "xnli_tr",
    "xnli_ur",
    "xnli_vi",
    "xnli_zh",
]

# ============================================================================
# STS TASK CATEGORIES
# ============================================================================

STS_TASKS = [
    "sts12",
    "sts22",
    "stsbenchmark",
]

# ============================================================================
# CLASSIFICATION TASK CATEGORIES
# ============================================================================

BINARY_CLASSIFICATION_TASKS = [
    "toxic_conversations",
    "amazon_counterfactual",
    "amazon_polarity",
    "imdb",
    "cola",
]

MULTIWAY_CLASSIFICATION_TASKS = [
    "banking77",
]

# ============================================================================
# CLUSTERING TASK CATEGORIES
# ============================================================================

CLUSTERING_TASKS = [
    "amazon_reviews",
    "emotion",
    "mtop_intent",
    "mtop_domain",
    "massive_scenario",
    "massive_intent",
    "tweet_sentiment",
    "arxiv_clustering_p2p",
    "arxiv_clustering_s2s",
    "biorxiv_clustering_p2p",
    "biorxiv_clustering_s2s",
    "medrxiv_clustering_p2p",
    "medrxiv_clustering_s2s",
    "reddit_clustering_p2p",
    "reddit_clustering_s2s",
    "stackexchange_clustering_p2p",
    "stackexchange_clustering_s2s",
    "twentynewsgroups",
]

# ============================================================================
# TASK TO CATEGORY MAPPING
# ============================================================================

# Create a mapping from task name to category
TASK_TO_CATEGORY = {}

# Retrieval tasks - organized by subcategory
for task in OPEN_DOMAIN_QA:
    TASK_TO_CATEGORY[task] = "retrieval/open_domain_qa"

for task in DOMAIN_SPECIFIC_QA:
    TASK_TO_CATEGORY[task] = "retrieval/domain_specific_qa"

for task in GENERAL_RETRIEVAL:
    TASK_TO_CATEGORY[task] = "retrieval/general_retrieval"

for task in FACT_VERIFICATION:
    TASK_TO_CATEGORY[task] = "retrieval/fact_verification"

for task in PARAPHRASE_DETECTION:
    TASK_TO_CATEGORY[task] = "retrieval/paraphrase_detection"

for task in SCIENTIFIC_DOC_RETRIEVAL:
    TASK_TO_CATEGORY[task] = "retrieval/scientific_doc_retrieval"

for task in SUMMARIZATION:
    TASK_TO_CATEGORY[task] = "retrieval/summarization"

# NLI tasks
for task in NLI_TASKS:
    TASK_TO_CATEGORY[task] = "nli"

for task in XNLI_TASKS:
    TASK_TO_CATEGORY[task] = "nli/xnli"

# STS tasks
for task in STS_TASKS:
    TASK_TO_CATEGORY[task] = "sts"

# Classification tasks
for task in BINARY_CLASSIFICATION_TASKS:
    TASK_TO_CATEGORY[task] = "classification/binary"

for task in MULTIWAY_CLASSIFICATION_TASKS:
    TASK_TO_CATEGORY[task] = "classification/multiway"

# Clustering tasks
for task in CLUSTERING_TASKS:
    TASK_TO_CATEGORY[task] = "clustering"


def get_task_category(task_name: str) -> str:
    """
    Get the category for a given task name.
    
    Args:
        task_name: Name of the task (lowercase)
    
    Returns:
        Category path (e.g., "retrieval/open_domain_qa", "nli", "sts"), or None if unknown
    """
    return TASK_TO_CATEGORY.get(task_name.lower(), None)


def get_category_path(task_name: str, base_path: str) -> str:
    """
    Get the full path for saving a task's dataset, organized by category.
    
    Args:
        task_name: Name of the task
        base_path: Base path for datasets (e.g., "./results/datasets_negatives/model_name")
    
    Returns:
        Full path including category subfolder(s) if applicable, otherwise base path with task name
    
    Examples:
        >>> get_category_path("msmarco", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/retrieval/general_retrieval/msmarco"
        
        >>> get_category_path("snli", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/nli/snli"
        
        >>> get_category_path("xnli_ar", "./results/datasets_negatives/qwen3_8b")
        "./results/datasets_negatives/qwen3_8b/nli/xnli/xnli_ar"
    """
    category = get_task_category(task_name)
    if category:
        return f"{base_path}/{category}/{task_name}", category
    else:
        # Unknown task, keep in flat structure
        return f"{base_path}/{task_name}", "category_unknown"
