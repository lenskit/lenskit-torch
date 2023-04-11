import logging

from csr import CSR
import numpy as np

from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st

from csr.test_utils import csrs

from lktorch.negatives import NegSampler

_log = logging.getLogger(__name__)


def matrices():
    return csrs(st.integers(10, 500), st.integers(50, 500), values=False)


@settings(suppress_health_check=[HealthCheck.data_too_large])
@given(matrices())
def test_uniform(mat: CSR):
    # make sure everything has negative items
    nnzs = mat.rowptrs[1:] - mat.rowptrs[:mat.nrows]
    nnz2 = nnzs * 2
    assume(np.all(nnz2 <= mat.ncols))
    _log.info('sampling from %s', mat)

    dense = mat.to_scipy().toarray()

    samp = NegSampler(mat)
    users = np.arange(0, mat.nrows)
    items = samp.sample(users)
    assert np.all(dense[users, items] == 0)


@settings(suppress_health_check=[HealthCheck.data_too_large])
@given(matrices())
def test_pop(mat: CSR):
    # make sure everything has negative items
    nnzs = mat.rowptrs[1:] - mat.rowptrs[:mat.nrows]
    nnz2 = nnzs * 2
    assume(np.all(nnz2 <= mat.ncols))

    _log.info('sampling from %s', mat)

    dense = mat.to_scipy().toarray()

    samp = NegSampler(mat, 'pop')
    users = np.arange(0, mat.nrows)
    items = samp.sample(users)
    assert np.all(dense[users, items] == 0)
