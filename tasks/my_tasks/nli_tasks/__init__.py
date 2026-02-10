"""
My NLI (Natural Language Inference) tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.xnli import (
    XNLIAr,
    XNLIBg,
    XNLIDe,
    XNLIEl,
    XNLIEs,
    XNLIFr,
    XNLIHi,
    XNLIRu,
    XNLISw,
    XNLITh,
    XNLITr,
    XNLIUr,
    XNLIVi,
    XNLIZh,
)

__all__ = [
    "XNLIAr",
    "XNLIBg",
    "XNLIDe",
    "XNLIEl",
    "XNLIEs",
    "XNLIFr",
    "XNLIHi",
    "XNLIRu",
    "XNLISw",
    "XNLITh",
    "XNLITr",
    "XNLIUr",
    "XNLIVi",
    "XNLIZh",
]
