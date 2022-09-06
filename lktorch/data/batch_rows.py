"""
Support for managing batches of rows, permuting and iterating them.
"""

import logging
import numpy as np
import math

import torch

_log = logging.getLogger(__name__)


class RowBatcher:
    """
    Manage batches of row indexes.

    Args:
        nrows(int):
            The number of rows.
        batch_size(int):
            The batch size.
        rng(numpy.random.Generator):
            Random number generator for permuting rows prior to each iteration pass.
    """
    def __init__(self, nrows, batch_size, rng):
        self.nrows = nrows
        self.batch_size = batch_size
        self._rng = rng
        self.permutation = np.arange(self.nrows, dtype='i4')

    @property
    def batch_count(self):
        """
        The number of batches in this batcher.
        """
        return math.ceil(self.nrows / self.batch_size)

    def batch(self, idx):
        """
        Fetch a batch by batch number.

        Args:
            idx(int): the index of the batch to fetch.

        Returns:
            numpy.ndarray:
                The row indices of this batch.
        """
        start = idx * self.batch_size
        if start >= self.nrows or start < 0:
            raise ValueError(f"invalid batch index {idx}")
        end = min(start + self.batch_size, self.nrows)
        picked = self.permutation[start:end]
        return picked

    def shuffle(self):
        """
        Re-shuffle the row permutation for the next iteration through the batches.
        """
        _log.info('re-shuffling %d rows', self.nrows)
        self.rng.shuffle(self.permutation)
