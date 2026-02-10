"""
KALM Retrieval tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.arxivqa import ArxivQA
from .data_loaders.ayadataset import AyaDataset
from .data_loaders.ccnews import CCNews
from .data_loaders.codefeedback import CodeFeedback
from .data_loaders.cqadupstack import CQADupStack
from .data_loaders.dbpediaentity import DBpediaEntity
from .data_loaders.esci import ESCI
from .data_loaders.expertqa import ExpertQA
from .data_loaders.gooaq import GooAQ
from .data_loaders.medi2bge import MEDI2BGE
from .data_loaders.mldr import MLDR
from .data_loaders.msmarcov2 import MSMARCOv2
from .data_loaders.openorca import OpenOrca
from .data_loaders.ragdataset12000 import RAGDataset12000
from .data_loaders.searchqa import SearchQA
from .data_loaders.treccovid import TRECCOVID
from .data_loaders.webgptcomparisons import WebGPTComparisons
from .data_loaders.yahooanswers import YahooAnswers

__all__ = [
    "ArxivQA",
    "AyaDataset",
    "CCNews",
    "CodeFeedback",
    "CQADupStack",
    "DBpediaEntity",
    "ESCI",
    "ExpertQA",
    "GooAQ",
    "MEDI2BGE",
    "MLDR",
    "MSMARCOv2",
    "OpenOrca",
    "RAGDataset12000",
    "SearchQA",
    "TRECCOVID",
    "WebGPTComparisons",
    "YahooAnswers",
]
