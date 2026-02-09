"""
Retrieval tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data.amazonqa import AmazonQA
from .data.arguana import Arguana
from .data.bioasq import BioASQ
from .data.cnndm import CNNDM
from .data.coliee import COLIEE
from .data.eli5 import ELI5
from .data.fever import FEVER
from .data.fiqa2018 import FiQA2018
from .data.hotpotqa import HotpotQA
from .data.miracl import MIRACL
from .data.mrtydi import MrTyDi
from .data.msmarco import MSMARCO
from .data.msmarcov2 import MSMARCOv2
from .data.naturalquestions import NaturalQuestions
from .data.nfcorpus import NFCorpus
from .data.paq import PAQ
from .data.pubmedqa import PubMedQA
from .data.qqp import QQP
from .data.s2orcabstractcitation import S2ORCAbstractCitation
from .data.s2orctitleabstract import S2ORCTitleAbstract
from .data.s2orctitlecitation import S2ORCTitleCitation
from .data.scifact import SciFact
from .data.sentencecompression import SentenceCompression
from .data.specter import SPECTER
from .data.squad import SQuAD
from .data.stackexchange import StackExchangeRetrieval
from .data.stackexchangedupquestionsp2p import StackExchangeDupQuestionsP2P
from .data.stackexchangedupquestionss2s import StackExchangeDupQuestionsS2S
from .data.stackoverflow_dup import StackOverflowDupQuestions
from .data.triviaqa import TriviaQA
from .data.xsum import XSum

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
