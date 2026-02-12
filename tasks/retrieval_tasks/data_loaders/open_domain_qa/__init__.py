"""Open Domain QA tasks."""

from .naturalquestions import NaturalQuestions
from .triviaqa import TriviaQA
from .paq import PAQ
from .eli5 import ELI5
from .squad import SQuAD
from .hotpotqa import HotpotQA

__all__ = [
    "NaturalQuestions",
    "TriviaQA",
    "PAQ",
    "ELI5",
    "SQuAD",
    "HotpotQA",
]
