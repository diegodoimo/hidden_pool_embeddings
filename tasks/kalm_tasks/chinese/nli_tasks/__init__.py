"""
Chinese NLI (Natural Language Inference) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.cail2019scm import CAIL2019SCM
from .data_loaders.cmnli import CMNLI
from .data_loaders.nlizh import NLIZh
from .data_loaders.ocnli import OCNLI
from .data_loaders.xnlizh import XNLIZh

__all__ = [
    "CAIL2019SCM",
    "CMNLI",
    "NLIZh",
    "OCNLI",
    "XNLIZh",
]
