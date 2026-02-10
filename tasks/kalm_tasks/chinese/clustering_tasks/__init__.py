"""
Chinese Clustering tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.cslclustering import CSLClustering
from .data_loaders.iflytekglustering import IFlyTekClustering
from .data_loaders.jdreviewclustering import JDReviewClustering
from .data_loaders.onlineshoppingclustering import OnlineShoppingClustering
from .data_loaders.thucnewsclustering import THUCNewsClustering
from .data_loaders.tnewsclustering import TNewsClustering
from .data_loaders.waimaiclustering import WaimaiClustering

__all__ = [
    "CSLClustering",
    "IFlyTekClustering",
    "JDReviewClustering",
    "OnlineShoppingClustering",
    "THUCNewsClustering",
    "TNewsClustering",
    "WaimaiClustering",
]
