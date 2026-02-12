"""Scientific Document Retrieval tasks."""

from .s2orcabstractcitation import S2ORCAbstractCitation
from .s2orctitleabstract import S2ORCTitleAbstract
from .s2orctitlecitation import S2ORCTitleCitation
from .specter import SPECTER

__all__ = [
    "S2ORCAbstractCitation",
    "S2ORCTitleAbstract",
    "S2ORCTitleCitation",
    "SPECTER",
]
