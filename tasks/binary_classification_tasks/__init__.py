"""
Binary classification tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.amazoncounterfactualclassification import (
    AmazonCounterfactualClassification,
)
from .data_loaders.amazonpolarityclassification import AmazonPolarityClassification
from .data_loaders.colaclassification import ColaClassification
from .data_loaders.imdbclassification import ImdbClassification
from .data_loaders.toxicconversations50k import ToxicConversations50k

__all__ = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "ColaClassification",
    "ImdbClassification",
    "ToxicConversations50k",
]
