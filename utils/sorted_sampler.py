import math
from collections.abc import Iterator
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np

__all__ = ["DistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class LenghtSortedSampler(Sampler[_T_co]):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

        # indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        lengths = [len(instance) for instance in self.dataset["input_ids"]]
        indices = list(np.argsort(lengths))[::-1]
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)

        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        self.indices = indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[_T_co]:

        if len(self.indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(self.indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore [bad-return]
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples
