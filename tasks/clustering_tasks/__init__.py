"""
Clustering tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data.amazonreviewsclustering import AmazonReviewsClustering
from .data.arxivclusteringp2p import ArxivClusteringP2P
from .data.arxivclusterings2s import ArxivClusteringS2S
from .data.biorxivclusteringp2p import BiorxivClusteringP2P
from .data.biorxivclusterings2s import BiorxivClusteringS2S
from .data.emotionclustering import EmotionClustering
from .data.massiveintentclustering import MassiveIntentClustering
from .data.massivescenarioclustering import MassiveScenarioClustering
from .data.medrxivclusteringp2p import MedrxivClusteringP2P
from .data.medrxivclusterings2s import MedrxivClusteringS2S
from .data.mtopdomainclustering import MTOPDomainClustering
from .data.mtopintentclustering import MTOPIntentClustering
from .data.redditclusteringp2p import RedditClusteringP2P
from .data.redditclusterings2s import RedditClusteringS2S
from .data.stackexchangeclusteringp2p import StackExchangeClusteringP2P
from .data.stackexchangeclusterings2s import StackExchangeClusteringS2S
from .data.tweetsentimentextractionclustering import TweetSentimentExtractionClustering
from .data.twentynewsgroupsclustering import TwentyNewsgroupsClustering

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
