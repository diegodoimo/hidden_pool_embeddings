"""
KALM Clustering tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.multilingualsentimentclustering import MultilingualSentimentClustering

__all__ = [
    "MultilingualSentimentClustering",
]
