"""
Classification tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data.amazoncounterfactualclassification import AmazonCounterfactualClassification
from .data.amazonpolarityclassification import AmazonPolarityClassification
from .data.banking77classification import Banking77Classification
from .data.colaclassification import ColaClassification
from .data.dbpediaclassification import DBPediaClassification
from .data.imdbclassification import ImdbClassification
from .data.toxicconversations50k import ToxicConversations50k

__all__ = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "Banking77Classification",
    "ColaClassification",
    "DBPediaClassification",
    "ImdbClassification",
    "ToxicConversations50k",
]
