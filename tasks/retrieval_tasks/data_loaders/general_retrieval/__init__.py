"""General Retrieval tasks."""

from .msmarco import MSMARCO
from .nfcorpus import NFCorpus
from .stackexchange import StackExchangeRetrieval
from .miracl import MIRACL
from .mrtydi import MrTyDi
from .arguana import Arguana

__all__ = [
    "MSMARCO",
    "NFCorpus",
    "StackExchangeRetrieval",
    "MIRACL",
    "MrTyDi",
    "Arguana",
]
