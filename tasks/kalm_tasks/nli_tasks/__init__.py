"""
KALM NLI (Natural Language Inference) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.all_nli import ALL_NLI
from .data_loaders.contractnli import ContractNLI
from .data_loaders.simcse import SimCSENLI

__all__ = [
    "ALL_NLI",
    "ContractNLI",
    "SimCSENLI",
]
