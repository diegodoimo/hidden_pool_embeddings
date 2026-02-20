from .binary_classification_tasks import *
from .classification_tasks import *
from .clustering_tasks import *
from .nli_tasks import *
from .retrieval_tasks import *
from .sts_tasks import *


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


# Define task categorization based on their characteristics
# NLI tasks are those with "nli" in the name
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


def get_task(name: str):
    if name not in NAME_TO_TASK:
        raise ValueError(
            f"Unknown task '{name}'. Available tasks: {list(NAME_TO_TASK)}"
        )

    task = NAME_TO_TASK[name]

    return task
