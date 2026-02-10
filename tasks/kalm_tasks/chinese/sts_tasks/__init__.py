"""
Chinese STS (Semantic Textual Similarity) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.afqmc import AFQMC
from .data_loaders.atec import ATEC
from .data_loaders.bq import BQ
from .data_loaders.chinesests import ChineseSTS
from .data_loaders.cinlid import CINLID
from .data_loaders.qbqtc import QBQTC
from .data_loaders.simclue import SimCLUE

__all__ = [
    "AFQMC",
    "ATEC",
    "BQ",
    "ChineseSTS",
    "CINLID",
    "QBQTC",
    "SimCLUE",
]
