"""
NLI (Natural Language Inference) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.all_nli import ALL_NLI
from .data_loaders.anli import ANLI
from .data_loaders.mnli import MNLI
from .data_loaders.snli import SNLI

__all__ = [
    "ALL_NLI",
    "ANLI",
    "MNLI",
    "SNLI",
]
