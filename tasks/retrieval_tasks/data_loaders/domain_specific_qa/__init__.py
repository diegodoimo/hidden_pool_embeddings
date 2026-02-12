"""Domain-Specific QA tasks."""

from .bioasq import BioASQ
from .pubmedqa import PubMedQA
from .fiqa2018 import FiQA2018
from .amazonqa import AmazonQA
from .coliee import COLIEE

__all__ = [
    "BioASQ",
    "PubMedQA",
    "FiQA2018",
    "AmazonQA",
    "COLIEE",
]
