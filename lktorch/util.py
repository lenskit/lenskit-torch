import torch
from numba import njit


def torch_dot(x, y):
    """
    Batch-wise dot products.  PyTorch doesn't have these natively for some reason;
    this `forum post`_ documents them.

    .. _`forum post`: https://discuss.pytorch.org/t/dot-product-batch-wise/9746/3
    """
    # torch doesn't have this
    # derived from https://discuss.pytorch.org/t/dot-product-batch-wise/9746/3
    B, S = x.shape
    return torch.bmm(x.view(B, 1, S), y.view(B, S, 1)).reshape(-1)
