import numpy as np
from csr import CSR

from . import TrainingStrategy, TrainBatch

class Observed(TrainingStrategy):
    ratings: CSR
    user_ids: np.ndarray

    def setup(self, ratings: CSR):
        self.ratings = ratings
        self.user_ids = ratings.rowinds()

    def n_rows(self):
        return self.ratings.nnz

    def row_size(self):
        return 1

    def train_batch(self, rows):
        users = self.user_ids[rows]
        items = self.ratings.colinds[rows]
        if self.ratings.values is None:
            values = None
        else:
            values = self.ratings.values[rows]

        return TrainBatch(users, items, values)
