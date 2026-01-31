from .classification_tasks import *
from .retrieval_tasks import *
from .sts_tasks import *


NAME_TO_TASK = {
    # MTEB-style retrieval tasks
    "msmarco": MSMARCO,
    "msmarco-v2": MSMARCOv2,
    "hotpotqa": HotpotQA,
    "fever": FEVER,
    "nfcorpus": NFCorpus,
    "fiqa2018": FiQA2018,
    "mrtydi": MrTyDi,
    "scifact": SciFact,
    # Sentence-transformers style retrieval tasks
    "naturalquestions": NaturalQuestions,
    "all_nli": ALL_NLI,
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
    # NLI tasks
    "arguana": Arguana,
    "snli": SNLI,
    "mnli": MNLI,
    "anli": ANLI,
    # QA tasks with custom loaders
    "squad": SQuAD,
    "stackexchange": StackExchangeRetrieval,
    "bioasq": BioASQ,
    "miracl": MIRACL,
    "pubmedqa": PubMedQA,
    # Summarization tasks
    "xsum": XSum,
    "cnndm": CNNDM,
    # Reranking tasks
    "stackoverflow_dup": StackOverflowDupQuestions,
    # STS tasks
    "sts12": STS12,
    "sts22": STS22,
    "stsbenchmark": STSBenchmark,
    # Classification tasks
    "dbpedia": DBPediaClassification,
    "toxic_conversations": ToxicConversations50k,
    "banking77": Banking77Classification,
    "reddit_clustering": RedditClusteringP2P,
}


def get_task(name: str):
    if name not in NAME_TO_TASK:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {list(NAME_TO_TASK)}")

    task = NAME_TO_TASK[name]

    return task
