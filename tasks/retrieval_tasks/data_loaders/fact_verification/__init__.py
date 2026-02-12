"""Fact Verification tasks."""

from .fever import FEVER
from .scifact import SciFact

__all__ = [
    "FEVER",
    "SciFact",
]
