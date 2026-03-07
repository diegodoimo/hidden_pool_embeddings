import os

from .binary_classification_tasks import *
from .classification_tasks import *
from .clustering_tasks import *
from .nli_tasks import *
from .retrieval_tasks import *
from .sts_tasks import *


def task_name_to_inner_path(task_name: str) -> str:
    """Build canonical inner path from NAME_TO_TASK_SUBTASK_PATH.

    Returns e.g. "retrieval/general_retrieval/arguana" or "nli/snli".
    """
    info = NAME_TO_TASK_SUBTASK_PATH[task_name]
    parent = info["parent_folder"]
    subparent = info["subparent_folder"]
    if subparent is not None:
        return os.path.join(parent, subparent, task_name)
    return os.path.join(parent, task_name)


NAME_TO_TASK = {
    # MTEB-style retrieval tasks
    "msmarco": MSMARCO,
    "hotpotqa": HotpotQA,
    "fever": FEVER,
    "nfcorpus": NFCorpus,
    "fiqa2018": FiQA2018,
    "mrtydi": MrTyDi,
    "scifact": SciFact,
    # Sentence-transformers style retrieval tasks
    "naturalquestions": NaturalQuestions,
    "paq": PAQ,
    "eli5": ELI5,
    "triviaqa": TriviaQA,
    "coliee": COLIEE,
    "s2orc_title_abstract": S2ORCTitleAbstract,
    "s2orc_title_citation": S2ORCTitleCitation,
    "s2orc_abstract_citation": S2ORCAbstractCitation,
    "specter": SPECTER,
    "sentence_compression": SentenceCompression,
    "stackexchange_dup_s2s": StackExchangeDupQuestionsS2S,
    "stackexchange_dup_p2p": StackExchangeDupQuestionsP2P,
    "qqp": QQP,
    # Argument retrieval tasks
    "arguana": Arguana,
    # NLI tasks
    "snli": SNLI,
    "mnli": MNLI,
    "anli": ANLI,
    "xnli": XNLI,
    # QA tasks with custom loaders
    "squad": SQuAD,
    "stackexchange": StackExchangeRetrieval,
    "bioasq": BioASQ,
    "miracl": MIRACL,
    "pubmedqa": PubMedQA,
    "amazonqa": AmazonQA,
    "gooaq": GooAQ,
    "yahooanswers": YahooAnswers,
    # Summarization tasks
    "xsum": XSum,
    "cnndm": CNNDM,
    "wikihow": WikiHow,
    # Reranking tasks
    "stackoverflow_dup": StackOverflowDupQuestions,
    # STS tasks
    "sts12": STS12,
    "sts22": STS22,
    "stsbenchmark": STSBenchmark,
    # Binary Classification tasks
    "toxic_conversations": ToxicConversations50k,
    "amazon_counterfactual": AmazonCounterfactualClassification,
    "amazon_polarity": AmazonPolarityClassification,
    "imdb": ImdbClassification,
    "cola": ColaClassification,
    # Multi-way Classification tasks
    "banking77": Banking77Classification,
    # Clustering tasks
    "amazon_reviews": AmazonReviewsClustering,
    "emotion": EmotionClustering,
    "mtop_intent": MTOPIntentClustering,
    "mtop_domain": MTOPDomainClustering,
    "massive_scenario": MassiveScenarioClustering,
    "massive_intent": MassiveIntentClustering,
    "tweet_sentiment": TweetSentimentExtractionClustering,
    "arxiv_clustering_p2p": ArxivClusteringP2P,
    "arxiv_clustering_s2s": ArxivClusteringS2S,
    "biorxiv_clustering_p2p": BiorxivClusteringP2P,
    "biorxiv_clustering_s2s": BiorxivClusteringS2S,
    "medrxiv_clustering_p2p": MedrxivClusteringP2P,
    "medrxiv_clustering_s2s": MedrxivClusteringS2S,
    "reddit_clustering_p2p": RedditClusteringP2P,
    "reddit_clustering_s2s": RedditClusteringS2S,
    "stackexchange_clustering_p2p": StackExchangeClusteringP2P,
    "stackexchange_clustering_s2s": StackExchangeClusteringS2S,
    "twentynewsgroups": TwentyNewsgroupsClustering,
}


TRANSLATE_F2LLM_NAME = {
    # identical names
    "amazon_counterfactual": "amazon_counterfactual",
    "amazon_polarity": "amazon_polarity",
    "amazon_reviews": "amazon_reviews",
    "anli": "anli",
    "arguana": "arguana",
    "arxiv_clustering_p2p": "arxiv_clustering_p2p",
    "arxiv_clustering_s2s": "arxiv_clustering_s2s",
    "banking77": "banking77",
    "bioasq": "bioasq",
    "biorxiv_clustering_p2p": "biorxiv_clustering_p2p",
    "biorxiv_clustering_s2s": "biorxiv_clustering_s2s",
    "cola": "cola",
    "coliee": "coliee",
    "eli5": "eli5",
    "emotion": "emotion",
    "fever": "fever",
    "hotpotqa": "hotpotqa",
    "imdb": "imdb",
    "massive_intent": "massive_intent",
    "massive_scenario": "massive_scenario",
    "medrxiv_clustering_p2p": "medrxiv_clustering_p2p",
    "medrxiv_clustering_s2s": "medrxiv_clustering_s2s",
    "miracl": "miracl",
    "mnli": "mnli",
    "msmarco": "msmarco",
    "mtop_domain": "mtop_domain",
    "mtop_intent": "mtop_intent",
    "nfcorpus": "nfcorpus",
    "pubmedqa": "pubmedqa",
    "qqp": "qqp",
    "reddit_clustering_p2p": "reddit_clustering_p2p",
    "reddit_clustering_s2s": "reddit_clustering_s2s",
    "s2orc_abstract_citation": "s2orc_abstract_citation",
    "s2orc_title_abstract": "s2orc_title_abstract",
    "s2orc_title_citation": "s2orc_title_citation",
    "scifact": "scifact",
    "sentence_compression": "sentence_compression",
    "snli": "snli",
    "specter": "specter",
    "squad": "squad",
    "paq": "paq",
    "stackexchange": "stackexchange",
    "stackexchange_clustering_p2p": "stackexchange_clustering_p2p",
    "stackexchange_clustering_s2s": "stackexchange_clustering_s2s",
    "sts12": "sts12",
    "sts22": "sts22",
    "stsbenchmark": "stsbenchmark",
    "toxic_conversations": "toxic_conversations",
    "triviaqa": "triviaqa",
    "twentynewsgroups": "twentynewsgroups",
    "xsum": "xsum",
    # different names
    "amazon_qa": "amazonqa",
    "cnn_dm": "cnndm",
    "fiqa": "fiqa2018",
    "mrtidy": "mrtydi",
    "natural_questions": "naturalquestions",
    "paq_part2": "paq",
    "stackexchange_part2": "stackexchange",
    "stackexchangedupquestions_p2p": "stackexchange_dup_p2p",
    "stackexchangedupquestions_s2s": "stackexchange_dup_s2s",
    "stackoverflowdupquestions": "stackoverflow_dup",
    "tweet_sentiment_extraction": "tweet_sentiment",
}


NAME_TO_TASK_TYPE = {
    # Retrieval tasks
    "msmarco": "Retrieval",
    "hotpotqa": "Retrieval",
    "fever": "Retrieval",
    "nfcorpus": "Retrieval",
    "fiqa2018": "Retrieval",
    "mrtydi": "Retrieval",
    "scifact": "Retrieval",
    "naturalquestions": "Retrieval",
    "paq": "Retrieval",
    "eli5": "Retrieval",
    "triviaqa": "Retrieval",
    "coliee": "Retrieval",
    "s2orc_title_abstract": "Retrieval",
    "s2orc_title_citation": "Retrieval",
    "s2orc_abstract_citation": "Retrieval",
    "specter": "Retrieval",
    "sentence_compression": "Retrieval",
    "stackexchange_dup_s2s": "Retrieval",
    "stackexchange_dup_p2p": "Retrieval",
    "qqp": "Retrieval",
    "arguana": "Retrieval",
    "squad": "Retrieval",
    "stackexchange": "Retrieval",
    "bioasq": "Retrieval",
    "miracl": "Retrieval",
    "pubmedqa": "Retrieval",
    "amazonqa": "Retrieval",
    "gooaq": "Retrieval",
    "yahooanswers": "Retrieval",
    # NLI tasks -> PairClassification
    "snli": "PairClassification",
    "mnli": "PairClassification",
    "anli": "PairClassification",
    "xnli": "PairClassification",
    # Summarization tasks
    "xsum": "Summarization",
    "cnndm": "Summarization",
    "wikihow": "Summarization",
    # Reranking tasks
    "stackoverflow_dup": "Reranking",
    # STS tasks
    "sts12": "STS",
    "sts22": "STS",
    "stsbenchmark": "STS",
    # Classification tasks
    "toxic_conversations": "Classification",
    "amazon_counterfactual": "Classification",
    "amazon_polarity": "Classification",
    "imdb": "Classification",
    "cola": "Classification",
    "banking77": "Classification",
    # Clustering tasks
    "amazon_reviews": "Clustering",
    "emotion": "Clustering",
    "mtop_intent": "Clustering",
    "mtop_domain": "Clustering",
    "massive_scenario": "Clustering",
    "massive_intent": "Clustering",
    "tweet_sentiment": "Clustering",
    "arxiv_clustering_p2p": "Clustering",
    "arxiv_clustering_s2s": "Clustering",
    "biorxiv_clustering_p2p": "Clustering",
    "biorxiv_clustering_s2s": "Clustering",
    "medrxiv_clustering_p2p": "Clustering",
    "medrxiv_clustering_s2s": "Clustering",
    "reddit_clustering_p2p": "Clustering",
    "reddit_clustering_s2s": "Clustering",
    "stackexchange_clustering_p2p": "Clustering",
    "stackexchange_clustering_s2s": "Clustering",
    "twentynewsgroups": "Clustering",
}

assert set(NAME_TO_TASK.keys()) == set(NAME_TO_TASK_TYPE.keys())

RETRIEVAL_SUBSET = {task for task, task_type in NAME_TO_TASK_TYPE.items() if task_type not in ["Clustering", "Classification"]}

NAME_TO_TASK_SUBTASK_PATH = {
    # Retrieval - general_retrieval
    "msmarco": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    "hotpotqa": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "fever": {"parent_folder": "retrieval", "subparent_folder": "fact_verification"},
    "nfcorpus": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    "fiqa2018": {"parent_folder": "retrieval", "subparent_folder": "domain_specific_qa"},
    "mrtydi": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    "scifact": {"parent_folder": "retrieval", "subparent_folder": "fact_verification"},
    "arguana": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    "stackexchange": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    "miracl": {"parent_folder": "retrieval", "subparent_folder": "general_retrieval"},
    # Retrieval - domain_specific_qa
    "naturalquestions": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "paq": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "eli5": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "triviaqa": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "coliee": {"parent_folder": "retrieval", "subparent_folder": "domain_specific_qa"},
    "s2orc_title_abstract": {"parent_folder": "retrieval", "subparent_folder": "scientific_doc_retrieval"},
    "s2orc_title_citation": {"parent_folder": "retrieval", "subparent_folder": "scientific_doc_retrieval"},
    "s2orc_abstract_citation": {"parent_folder": "retrieval", "subparent_folder": "scientific_doc_retrieval"},
    "specter": {"parent_folder": "retrieval", "subparent_folder": "scientific_doc_retrieval"},
    "sentence_compression": {"parent_folder": "retrieval", "subparent_folder": "summarization"},
    "stackexchange_dup_s2s": {"parent_folder": "retrieval", "subparent_folder": "paraphrase_detection"},
    "stackexchange_dup_p2p": {"parent_folder": "retrieval", "subparent_folder": "paraphrase_detection"},
    "qqp": {"parent_folder": "retrieval", "subparent_folder": "paraphrase_detection"},
    # Retrieval - QA with custom loaders
    "squad": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "bioasq": {"parent_folder": "retrieval", "subparent_folder": "domain_specific_qa"},
    "pubmedqa": {"parent_folder": "retrieval", "subparent_folder": "domain_specific_qa"},
    "amazonqa": {"parent_folder": "retrieval", "subparent_folder": "domain_specific_qa"},
    "gooaq": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    "yahooanswers": {"parent_folder": "retrieval", "subparent_folder": "open_domain_qa"},
    # Retrieval - summarization
    "xsum": {"parent_folder": "retrieval", "subparent_folder": "summarization"},
    "cnndm": {"parent_folder": "retrieval", "subparent_folder": "summarization"},
    "wikihow": {"parent_folder": "retrieval", "subparent_folder": "summarization"},
    # Retrieval - reranking
    "stackoverflow_dup": {"parent_folder": "retrieval", "subparent_folder": "paraphrase_detection"},
    # NLI tasks
    "snli": {"parent_folder": "nli", "subparent_folder": None},
    "mnli": {"parent_folder": "nli", "subparent_folder": None},
    "anli": {"parent_folder": "nli", "subparent_folder": None},
    "xnli": {"parent_folder": "nli", "subparent_folder": None},
    # STS tasks
    "sts12": {"parent_folder": "sts", "subparent_folder": None},
    "sts22": {"parent_folder": "sts", "subparent_folder": None},
    "stsbenchmark": {"parent_folder": "sts", "subparent_folder": None},
    # Binary classification tasks
    "toxic_conversations": {"parent_folder": "binary_classification", "subparent_folder": None},
    "amazon_counterfactual": {"parent_folder": "binary_classification", "subparent_folder": None},
    "amazon_polarity": {"parent_folder": "binary_classification", "subparent_folder": None},
    "imdb": {"parent_folder": "binary_classification", "subparent_folder": None},
    "cola": {"parent_folder": "binary_classification", "subparent_folder": None},
    # Multi-way classification tasks
    "banking77": {"parent_folder": "classification", "subparent_folder": None},
    # Clustering tasks
    "amazon_reviews": {"parent_folder": "clustering", "subparent_folder": None},
    "emotion": {"parent_folder": "clustering", "subparent_folder": None},
    "mtop_intent": {"parent_folder": "clustering", "subparent_folder": None},
    "mtop_domain": {"parent_folder": "clustering", "subparent_folder": None},
    "massive_scenario": {"parent_folder": "clustering", "subparent_folder": None},
    "massive_intent": {"parent_folder": "clustering", "subparent_folder": None},
    "tweet_sentiment": {"parent_folder": "clustering", "subparent_folder": None},
    "arxiv_clustering_p2p": {"parent_folder": "clustering", "subparent_folder": None},
    "arxiv_clustering_s2s": {"parent_folder": "clustering", "subparent_folder": None},
    "biorxiv_clustering_p2p": {"parent_folder": "clustering", "subparent_folder": None},
    "biorxiv_clustering_s2s": {"parent_folder": "clustering", "subparent_folder": None},
    "medrxiv_clustering_p2p": {"parent_folder": "clustering", "subparent_folder": None},
    "medrxiv_clustering_s2s": {"parent_folder": "clustering", "subparent_folder": None},
    "reddit_clustering_p2p": {"parent_folder": "clustering", "subparent_folder": None},
    "reddit_clustering_s2s": {"parent_folder": "clustering", "subparent_folder": None},
    "stackexchange_clustering_p2p": {"parent_folder": "clustering", "subparent_folder": None},
    "stackexchange_clustering_s2s": {"parent_folder": "clustering", "subparent_folder": None},
    "twentynewsgroups": {"parent_folder": "clustering", "subparent_folder": None},
}

assert set(NAME_TO_TASK_SUBTASK_PATH.keys()) == set(NAME_TO_TASK.keys())









# NLI tasks
NLI_TASKS = [
    "snli",
    "mnli",
    "anli",
    "xnli",
]

# STS tasks are semantic textual similarity tasks
STS_TASKS = ["sts12", "sts22", "stsbenchmark"]

# Binary Classification tasks
BINARY_CLASSIFICATION_TASKS = [
    "toxic_conversations",
    "amazon_counterfactual",
    "amazon_polarity",
    "imdb",
    "cola",
]

# Multi-way Classification tasks
CLASSIFICATION_TASKS = [
    "banking77",
]

# Clustering tasks
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


# Datasets sorted by total size (unique_queries + unique_documents), shortest first
# as they appear in dataset_list.txt
DATASETS_BY_SIZE = [
    # "sts12",  #      2,712
    # "sts22",  #      3,062
    # "scifact",  #      5,992
    # "stsbenchmark",  #      6,420
    # "nfcorpus",  #      6,547
    # "arguana",  #     28,728
    # "fiqa2018",  #     63,638
    # "squad",  #    118,846
    # "triviaqa",  #    137,483
    # "anli",  #    164,019
    # "qqp",  #    172,242
    # "naturalquestions",  #    175,446
    # #"miracl",  #    227,090               NOT DONE
    # "specter",  #    241,084
    # "sentence_compression",  #    356,409
    # "stackoverflow_dup",  #    373,863
    # "stackexchange_dup_p2p",  #    374,049
    # "wikihow",  #    422,657
    # "xsum",  #    429,542
    # "stackexchange_dup_s2s",  #    462,315
    # "mnli",  #    544,416
    # "pubmedqa",  #    546,761
    # "cnndm",  #    592,859
    # "snli",  #    637,407
    # "eli5",  #    648,429
    # "mrtydi",  #  1,051,025
    # "yahooanswers",  #  2,369,859
    # "amazonqa",  #  3,264,575
    "gooaq",  #  4,488,520
    "hotpotqa",  #  5,323,776
    "fever",  #  5,533,044
    "xnli",  #  7,253,838
    "msmarco",  #  9,351,742
    "stackexchange",  #  9,500,045
    # "bioasq",  # 27,284,021
    # "s2orc_abstract_citation",  # 56,386,121
    # "s2orc_title_citation",  # 65,572,442
    # "paq",  # 73,376,018
    # "s2orc_title_abstract",  # 82,561,651
]


def get_task(name: str):
    if name not in NAME_TO_TASK:
        raise ValueError(
            f"Unknown task '{name}'. Available tasks: {list(NAME_TO_TASK)}"
        )

    task = NAME_TO_TASK[name]

    return task
