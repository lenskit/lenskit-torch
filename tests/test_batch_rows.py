"""
Test the row-batch capabilities.
"""

from lktorch.data.batch import RowBatcher

from hypothesis import given
from hypothesis import strategies as st

from seedbank import numpy_rng

import numpy as np


def test_known_batch_count():
    "Test batch counts for some known sizes."
    rng = numpy_rng()
    rbs = RowBatcher(100, 10, rng)
    assert rbs.batch_count == 10

    rbs = RowBatcher(95, 10, rng)
    assert rbs.batch_count == 10

    rbs = RowBatcher(101, 10, rng)
    assert rbs.batch_count == 11


@given(st.integers(0, 10000), st.integers(1, 5000))
def test_batch_count(nrows, batch_size):
    "Test a wide range of batch counts"
    rbs = RowBatcher(nrows, batch_size, numpy_rng())

    assert rbs.nrows == nrows
    assert rbs.batch_size == batch_size

    nb = rbs.batch_count
    assert nb * batch_size >= nrows
    assert (nrows - nb * batch_size) < batch_size


@given(st.integers(1, 10000), st.integers(1, 5000))
def test_batch_rows(nrows, batch_size):
    "Test that we can get batch rows"
    rbs = RowBatcher(nrows, batch_size, numpy_rng())

    arrs = []
    for i in range(rbs.batch_count):
        arrs.append(rbs.batch(i))

    arr = np.concatenate(arrs)
    arr = np.unique(arr)
    assert len(arr) == nrows
    assert arr[0] == 0
    assert arr[-1] == nrows - 1

@given(st.integers(1, 10000), st.integers(1, 5000))
def test_batch_iter(nrows, batch_size):
    "Test that we can iterate batch rows"
    rbs = RowBatcher(nrows, batch_size, numpy_rng())

    arrs = []
    for batch in rbs:
        arrs.append(batch)

    arr = np.concatenate(arrs)
    arr = np.unique(arr)
    assert len(arr) == nrows
    assert arr[0] == 0
    assert arr[-1] == nrows - 1
