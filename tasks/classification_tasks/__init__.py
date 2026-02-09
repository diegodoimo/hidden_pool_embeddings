"""
Classification tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.amazoncounterfactualclassification import (
    AmazonCounterfactualClassification,
)
from .data_loaders.amazonpolarityclassification import AmazonPolarityClassification
from .data_loaders.banking77classification import Banking77Classification
from .data_loaders.colaclassification import ColaClassification
from .data_loaders.dbpediaclassification import DBPediaClassification
from .data_loaders.imdbclassification import ImdbClassification
from .data_loaders.toxicconversations50k import ToxicConversations50k

__all__ = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "Banking77Classification",
    "ColaClassification",
    "DBPediaClassification",
    "ImdbClassification",
    "ToxicConversations50k",
]
