"""
Clustering tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.amazonreviewsclustering import AmazonReviewsClustering
from .data_loaders.arxivclusteringp2p import ArxivClusteringP2P
from .data_loaders.arxivclusterings2s import ArxivClusteringS2S
from .data_loaders.biorxivclusteringp2p import BiorxivClusteringP2P
from .data_loaders.biorxivclusterings2s import BiorxivClusteringS2S
from .data_loaders.emotionclustering import EmotionClustering
from .data_loaders.massiveintentclustering import MassiveIntentClustering
from .data_loaders.massivescenarioclustering import MassiveScenarioClustering
from .data_loaders.medrxivclusteringp2p import MedrxivClusteringP2P
from .data_loaders.medrxivclusterings2s import MedrxivClusteringS2S
from .data_loaders.mtopdomainclustering import MTOPDomainClustering
from .data_loaders.mtopintentclustering import MTOPIntentClustering
from .data_loaders.redditclusteringp2p import RedditClusteringP2P
from .data_loaders.redditclusterings2s import RedditClusteringS2S
from .data_loaders.stackexchangeclusteringp2p import StackExchangeClusteringP2P
from .data_loaders.stackexchangeclusterings2s import StackExchangeClusteringS2S
from .data_loaders.tweetsentimentextractionclustering import (
    TweetSentimentExtractionClustering,
)
from .data_loaders.twentynewsgroupsclustering import TwentyNewsgroupsClustering

__all__ = [
    "AmazonReviewsClustering",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "EmotionClustering",
    "MassiveIntentClustering",
    "MassiveScenarioClustering",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "MTOPDomainClustering",
    "MTOPIntentClustering",
    "RedditClusteringP2P",
    "RedditClusteringS2S",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringS2S",
    "TweetSentimentExtractionClustering",
    "TwentyNewsgroupsClustering",
]
