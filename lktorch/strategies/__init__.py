from typing import NamedTuple, Optional
from abc import ABC, abstractmethod

import numpy as np
from csr import CSR
from torch import Tensor, from_numpy

class TrainBatch(NamedTuple):
    """
    A training batch.
    """

    "The user IDs for this batch."
    users: np.ndarray | Tensor
    "The item IDs for this batch."
    items: np.ndarray | Tensor
    "The values (if available) for the items in this batch."
    values: Optional[np.ndarray | Tensor]

    def to(self, dev=None):
        "Move the batch into PyTorch, and optionally to a device."
        args = []
        for field in self._fields:
            a = getattr(self, field)
            if isinstance(a, np.ndarray):
                a = from_numpy(a)
            if dev:
                a = a.to(dev)
            setattr(self, field, a)

        return self


class TrainingStrategy(ABC):
    """
    Interface for the strategies used for training LKTorch models.  A strategy
    encapsulates a means for obtaining a training batch, including negative
    examples as needed.  The strategy operates on a ratings matrix.

    Strategies operate in terms of NumPy ndarrays, which the caller is
    responsible for converting into tensors.
    """

    @abstractmethod
    def setup(self, ratings: CSR):
        """
        Set up the training model's internal structures for training.

        Args:
            ratings(csr.CSR):
                The ratings matrix in CSR form, with or without values.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_rows(self) -> int:
        """
        Get the number of training rows this strategy will present.  For
        userwise strategies, it will be the number of users; for pairwise and
        pointwise strategies, the number of pairs or observed points.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def row_size(self) -> int:
        """
        Get the size of each sample, in terms of the number of items that need
        to be scored.  For example, a pairwise strategy has a sample size of 2.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_batch(self, rows) -> TrainBatch:
        """
        Get a single training batch.

        Args:
            rows(numpy.ndarray):
                The indices of the rows to obtain (e.g. from a permutation).
                Must have size :math:`B` (the batch size).

        Returns:
            TrainBatch:
                The training batch.
        """
        raise NotImplementedError()
