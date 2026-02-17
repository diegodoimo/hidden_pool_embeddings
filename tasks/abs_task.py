from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass


class AbsTask(ABC):
    # Required class-level attributes
    hf_name: str
    hf_subset: Optional[str] = None
    split: str
    has_multiple_datasets: bool

    anchor_name: str
    positive_name: str
    negative_name: Optional[str] = None
    title_name: str = None
    qrels_name: str = None

    qrels_fields: Dict[str, str] = None
    anchor_fields: Dict[str, str] = None
    corpus_fields: Dict[str, str] = None
    subtasks: Optional[List] = None
    decontaminator: Optional[Callable] = None

    @classmethod
    @abstractmethod
    def validate_config(cls) -> None:
        """
        Validate that the dataset configuration is internally consistent.
        Should raise an exception if misconfigured.
        """
        pass


@dataclass
class TaskMetadata:
    type: str
    prompt: Optional[Dict[str, str]] = None
