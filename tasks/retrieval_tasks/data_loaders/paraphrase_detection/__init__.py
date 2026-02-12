"""Paraphrase Detection tasks."""

from .qqp import QQP
from .stackexchangedupquestionsp2p import StackExchangeDupQuestionsP2P
from .stackexchangedupquestionss2s import StackExchangeDupQuestionsS2S
from .stackoverflow_dup import StackOverflowDupQuestions

__all__ = [
    "QQP",
    "StackExchangeDupQuestionsP2P",
    "StackExchangeDupQuestionsS2S",
    "StackOverflowDupQuestions",
]
