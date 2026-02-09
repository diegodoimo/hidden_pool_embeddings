"""
Retrieval tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.amazonqa import AmazonQA
from .data_loaders.arguana import Arguana
from .data_loaders.bioasq import BioASQ
from .data_loaders.cnndm import CNNDM
from .data_loaders.coliee import COLIEE
from .data_loaders.eli5 import ELI5
from .data_loaders.fever import FEVER
from .data_loaders.fiqa2018 import FiQA2018
from .data_loaders.hotpotqa import HotpotQA
from .data_loaders.miracl import MIRACL
from .data_loaders.mrtydi import MrTyDi
from .data_loaders.msmarco import MSMARCO
from .data_loaders.msmarcov2 import MSMARCOv2
from .data_loaders.naturalquestions import NaturalQuestions
from .data_loaders.nfcorpus import NFCorpus
from .data_loaders.paq import PAQ
from .data_loaders.pubmedqa import PubMedQA
from .data_loaders.qqp import QQP
from .data_loaders.s2orcabstractcitation import S2ORCAbstractCitation
from .data_loaders.s2orctitleabstract import S2ORCTitleAbstract
from .data_loaders.s2orctitlecitation import S2ORCTitleCitation
from .data_loaders.scifact import SciFact
from .data_loaders.sentencecompression import SentenceCompression
from .data_loaders.specter import SPECTER
from .data_loaders.squad import SQuAD
from .data_loaders.stackexchange import StackExchangeRetrieval
from .data_loaders.stackexchangedupquestionsp2p import StackExchangeDupQuestionsP2P
from .data_loaders.stackexchangedupquestionss2s import StackExchangeDupQuestionsS2S
from .data_loaders.stackoverflow_dup import StackOverflowDupQuestions
from .data_loaders.triviaqa import TriviaQA
from .data_loaders.xsum import XSum

__all__ = [
    "AmazonQA",
    "Arguana",
    "BioASQ",
    "CNNDM",
    "COLIEE",
    "ELI5",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MIRACL",
    "MrTyDi",
    "MSMARCO",
    "MSMARCOv2",
    "NaturalQuestions",
    "NFCorpus",
    "PAQ",
    "PubMedQA",
    "QQP",
    "S2ORCAbstractCitation",
    "S2ORCTitleAbstract",
    "S2ORCTitleCitation",
    "SciFact",
    "SentenceCompression",
    "SPECTER",
    "SQuAD",
    "StackExchangeRetrieval",
    "StackExchangeDupQuestionsP2P",
    "StackExchangeDupQuestionsS2S",
    "StackOverflowDupQuestions",
    "TriviaQA",
    "XSum",
]
