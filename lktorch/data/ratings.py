"""
Support for storing & sampling ratings.
"""

from dataclasses import dataclass
from typing import NamedTuple
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import torch

from lenskit.data import sparse_ratings


class RatingBatch(NamedTuple):
    """
    A single batch of ratings.
    """
    users: NDArray[np.int32]
    items: NDArray[np.int32]
    vals: NDArray[np.float32]

    def to(self, device):
        return RatingBatch(
            self.users.to(device),
            self.items.to(device),
            self.vals.to(device)
        )


@dataclass
class RatingData:
    """
    Rating data for explicit-feedback models.
    """

    users: pd.Index
    items: pd.Index

    r_users: NDArray[np.int32]
    r_items: NDArray[np.int32]
    r_values: NDArray[np.float32]

    @classmethod
    def from_ratings(cls, ratings):
        """
        Convert a rating data frame to a rating data training object.

        Args:
            ratings(pandas.DataFrame):
                The rating data frame, with ``user``, ``item``, and ``rating`` columns.
        """

        data = sparse_ratings(ratings)
        users = np.require(data.matrix.rowinds(), 'i4')
        items = np.require(data.matrix.colinds, 'i4')
        values = data.matrix.values
        if values is None:
            values = np.ones(len(users), dtype='f4')
        else:
            values = np.require(values, 'f4')

        return cls(data.users, data.items, users, items, values)

    @property
    def n_samples(self):
        return len(self.r_users)

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    def batch_from_rows(self, rows):
        users = torch.from_numpy(self.r_users[rows])
        items = torch.from_numpy(self.r_items[rows])
        values = torch.from_numpy(self.r_values[rows])
        return RatingBatch(users, items, values)
