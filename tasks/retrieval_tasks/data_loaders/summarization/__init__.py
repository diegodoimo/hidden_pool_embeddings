"""Summarization tasks."""

from .cnndm import CNNDM
from .xsum import XSum
from .sentencecompression import SentenceCompression
from .wikihow import WikiHow

__all__ = [
    "CNNDM",
    "XSum",
    "SentenceCompression",
    "WikiHow",
]
