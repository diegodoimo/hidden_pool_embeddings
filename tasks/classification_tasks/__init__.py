"""
Multi-way classification tasks module.
Each task is defined in its own file with its associated loader.
"""

from .data_loaders.banking77classification import Banking77Classification

__all__ = [
    "Banking77Classification",
]
