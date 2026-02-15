"""General Retrieval tasks."""

from .msmarco import MSMARCO
from .nfcorpus import NFCorpus
from .stackexchange import StackExchangeRetrieval
from .miracl import MIRACL
from .mrtydi import (
    MrTyDiArabic,
    MrTyDiBengali,
    MrTyDiEnglish,
    MrTyDiFinnish,
    MrTyDiIndonesian,
    MrTyDiJapanese,
    MrTyDiKorean,
    MrTyDiRussian,
    MrTyDiSwahili,
    MrTyDiTelugu,
    MrTyDiThai,
)
from .arguana import Arguana

__all__ = [
    "MSMARCO",
    "NFCorpus",
    "StackExchangeRetrieval",
    "MIRACL",
    "MrTyDiArabic",
    "MrTyDiBengali",
    "MrTyDiEnglish",
    "MrTyDiFinnish",
    "MrTyDiIndonesian",
    "MrTyDiJapanese",
    "MrTyDiKorean",
    "MrTyDiRussian",
    "MrTyDiSwahili",
    "MrTyDiTelugu",
    "MrTyDiThai",
    "Arguana",
]
