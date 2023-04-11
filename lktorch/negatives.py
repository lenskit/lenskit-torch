"""
Utility code for sampling for PyTorch recommenders.
"""

from numba import njit

import numpy as np
from csr import CSR
from seedbank import numpy_rng


# @njit
def _sample_uniform(rng: np.random.Generator, mat: CSR, n):
    """
    Candidate sampling function for use with :py:func:`neg_sample`.
    It samples items uniformly at random.
    """

    return rng.integers(0, mat.ncols, n)


# @njit
def _sample_pop(rng: np.random.Generator, mat: CSR, n):
    """
    Candidate sampling function for use with :py:func:`neg_sample`.
    It samples items (approximately) proportionally to their popularity.
    """

    # we treat each item as having one extra rating for the purposes of
    # popularity-weighting, in order to allow sampling of items that never
    # appear in the matrix.
    j = rng.integers(0, mat.nnz + mat.ncols, n)
    cols = j - mat.nnz
    mask = cols < 0
    cols[mask] = mat.colinds[j[mask]]
    return cols


# @njit(nogil=True)
# def _neg_sample(rng: np.random.Generator, mat: CSR, uv, sample):
#     n = len(uv)
#     jv = np.empty(n, dtype=np.int32)
#     sc = np.ones(n, dtype=np.int32)

#     for i in range(n):
#         u = uv[i]
#         used = mat.row_cs(u)
#         j = sample(rng, mat)
#         while np.any(used == j):
#             j = sample(rng, mat)
#             sc[i] = sc[i] + 1
#         jv[i] = j

#     return jv, sc

@njit
def _ui_mask(mat: CSR, uv):
    n = len(uv)
    mask = np.zeros((n, mat.ncols), np.bool_)
    for r, u in enumerate(uv):
        sp, ep = mat.row_extent(u)
        for ci in range(sp, ep):
            c = mat.colinds[ci]
            mask[r, c] = True

    return mask


class NegSampler:
    """
    Sample negative examples (unrated items) for users, using a CSR to store the observed
    user-item interactions.  It uses rejection sampling to find items the user has not rated.

    Args:
        mat(csr.CSR):
            The user-item matrix. Its values are ignored and do not need to be present.
        weighting(str):
            The weighting to use. One of ``uniform`` or ``pop``.
        rng_spec:
            An RNG seed suitable to be passed to :func:`seedbank.numpy_rng`.
    """

    def __init__(self, mat: CSR, weighting='uniform', rng_spec=None):
        self.matrix = mat
        self.rng = numpy_rng(rng_spec)
        if weighting == 'uniform':
            self.sampler = _sample_uniform
        elif weighting == 'pop':
            self.sampler = _sample_pop

    def sample(self, users):
        """
        Sample negative item IDs for a list of users.

        Args:
            user(numpy.ndarray):
                The array of user IDs.

        Returns:
            numpy.ndarray:
                The array of item IDs.
        """
        n = len(users)
        mask = _ui_mask(self.matrix, users)
        rows = np.arange(n, dtype=np.int32)

        # initial sample
        js = self.sampler(self.rng, self.matrix, n)

        # check for bad items
        bad = mask[rows, js]
        nbad = np.sum(bad)
        total_bad = nbad
        while nbad:
            # we need to re-sample bad items
            js[bad] = self.sampler(self.rng, self.matrix, nbad)
            bad[bad] = mask[rows[bad], js[bad]]
            nbad = np.sum(bad)
            total_bad += nbad
            if total_bad >= n * 10:
                raise RuntimeError('trying too hard to sample negatives')

        # js, _cts = _neg_sample(self.rng, self.matrix, users, self.sampler)
        return js
