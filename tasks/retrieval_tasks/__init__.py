"""
Retrieval tasks module.
Each task is defined in its own file with its associated loader.
Tasks are organized into categories for better organization.
"""

# Open Domain QA
from .data_loaders.open_domain_qa import (
    NaturalQuestions,
    TriviaQA,
    PAQ,
    ELI5,
    SQuAD,
    HotpotQA,
    GooAQ,
    YahooAnswers,
)

# Domain-Specific QA
from .data_loaders.domain_specific_qa import (
    BioASQ,
    PubMedQA,
    FiQA2018,
    AmazonQA,
    COLIEE,
)

# General Retrieval
from .data_loaders.general_retrieval import (
    MSMARCO,
    NFCorpus,
    StackExchangeRetrieval,
    MIRACL,
    MrTyDi,
    Arguana,
)

# Fact Verification
from .data_loaders.fact_verification import (
    FEVER,
    SciFact,
)

# Paraphrase Detection
from .data_loaders.paraphrase_detection import (
    QQP,
    StackExchangeDupQuestionsP2P,
    StackExchangeDupQuestionsS2S,
    StackOverflowDupQuestions,
)

# Scientific Document Retrieval
from .data_loaders.scientific_doc_retrieval import (
    S2ORCAbstractCitation,
    S2ORCTitleAbstract,
    S2ORCTitleCitation,
    SPECTER,
)

# Summarization
from .data_loaders.summarization import (
    CNNDM,
    XSum,
    SentenceCompression,
    WikiHow,
)

__all__ = [
    # Open Domain QA
    "NaturalQuestions",
    "TriviaQA",
    "PAQ",
    "ELI5",
    "SQuAD",
    "HotpotQA",
    "GooAQ",
    "YahooAnswers",
    # Domain-Specific QA
    "BioASQ",
    "PubMedQA",
    "FiQA2018",
    "AmazonQA",
    "COLIEE",
    # General Retrieval
    "MSMARCO",
    "NFCorpus",
    "StackExchangeRetrieval",
    "MIRACL",
    "MrTyDi",
    "Arguana",
    # Fact Verification
    "FEVER",
    "SciFact",
    # Paraphrase Detection
    "QQP",
    "StackExchangeDupQuestionsP2P",
    "StackExchangeDupQuestionsS2S",
    "StackOverflowDupQuestions",
    # Scientific Document Retrieval
    "S2ORCAbstractCitation",
    "S2ORCTitleAbstract",
    "S2ORCTitleCitation",
    "SPECTER",
    # Summarization
    "CNNDM",
    "XSum",
    "SentenceCompression",
    "WikiHow",
]
