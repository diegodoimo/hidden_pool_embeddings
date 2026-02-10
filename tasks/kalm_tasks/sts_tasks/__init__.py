"""
KALM STS (Semantic Textual Similarity) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.nllb import NLLB
from .data_loaders.pawsxmultilingual import PAWSXMultilingual
from .data_loaders.quora import Quora
from .data_loaders.wikianswers import WikiAnswers

__all__ = [
    "NLLB",
    "PAWSXMultilingual",
    "Quora",
    "WikiAnswers",
]
