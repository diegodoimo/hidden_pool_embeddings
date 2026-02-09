"""
STS (Semantic Textual Similarity) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.sts12 import STS12
from .data_loaders.sts22 import STS22
from .data_loaders.stsbenchmark import STSBenchmark

__all__ = [
    "STS12",
    "STS22",
    "STSBenchmark",
]
