"""
STS (Semantic Textual Similarity) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data.sts12 import STS12
from .data.sts22 import STS22
from .data.stsbenchmark import STSBenchmark

__all__ = [
    "STS12",
    "STS22",
    "STSBenchmark",
]
