"""
NLI (Natural Language Inference) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data.all_nli import ALL_NLI
from .data.anli import ANLI
from .data.mnli import MNLI
from .data.snli import SNLI

__all__ = [
    "ALL_NLI",
    "ANLI",
    "MNLI",
    "SNLI",
]
