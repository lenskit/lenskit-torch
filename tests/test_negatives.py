from csr import CSR
import numpy as np

from hypothesis import given, assume
import hypothesis.strategies as st

from csr.test_utils import csrs

from lktorch.negatives import NegSampler

def matrices():
    return csrs(st.integers(10, 500), st.integers(50, 500), values=False)


@given(matrices())
def test_uniform(mat: CSR):
    # make sure everything has a negative item
    nnzs = mat.rowptrs[1:] - mat.rowptrs[:mat.nrows]
    nnz2 = nnzs * 2
    assume(np.all(nnz2 <= mat.ncols))
    print('sampling', mat)

    dense = mat.to_scipy().toarray()

    samp = NegSampler(mat)
    users = np.arange(0, mat.nrows)
    items = samp.sample(users)
    assert np.all(dense[users, items] == 0)
