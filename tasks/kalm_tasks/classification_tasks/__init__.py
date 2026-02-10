"""
KALM Classification tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.dbpediaclassification import DBPediaClassification

__all__ = [
    "DBPediaClassification",
]
