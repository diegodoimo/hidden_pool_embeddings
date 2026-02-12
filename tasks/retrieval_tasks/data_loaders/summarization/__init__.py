"""Summarization tasks."""

from .cnndm import CNNDM
from .xsum import XSum
from .sentencecompression import SentenceCompression

__all__ = [
    "CNNDM",
    "XSum",
    "SentenceCompression",
]
