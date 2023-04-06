import logging
from torch import nn, Tensor
from torch.linalg import vecdot

_log = logging.getLogger(__name__)


class MFNet(nn.Module):
    """
    Torch module that defines a basic matrix factorization model with
    user and item biases.

    See the :meth:`forward` method for a description of this module's
    inputs.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_feats(int): the embedding dimension
    """

    def __init__(self, n_users: int, n_items: int, n_feats: int):
        super().__init__()
        self.n_feats = n_feats
        self.n_users = n_users
        self.n_items = n_items

        # user and item bias terms
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)

        # user and item embeddings
        self.u_embed = nn.Embedding(n_users, n_feats)
        self.i_embed = nn.Embedding(n_items, n_feats)

        # rescale all initial values for better starting point
        # they started out as standard normals, those are pretty big
        self.u_bias.weight.data.mul_(0.05)
        self.i_bias.weight.data.mul_(0.05)
        self.u_embed.weight.data.mul_(0.05)
        self.i_embed.weight.data.mul_(0.05)


    def forward(self, users: Tensor, items: Tensor):
        """
        Forward pass for the matrix factorization module.

        Args:
            users(torch.Tensor):
                The user IDs.  During training, this tensor usually has shape
                ``(B, 1)`` (one user ID per batch row); during inference, it
                usually has dimension ``1``.  If a tensor of size ``B`` is
                passed, it is reshaped.
            items(torch.Tensor):
                The item IDs.  During training, its precise shape depends on the
                training regime; it is usually ``(B, n)``, where ``n`` is the
                number of items to score for each user in the batch.  In
                pairwise training, for example, ``n=2``.

        Returns:
            torch.Tensor:
                A tensor of the same shape as ``items``, with scores for each
                item in the batch.
        """
        # look up biases and embeddings
        # biases have dimension (N, 1); remove the 1 by reshaping to match user/item inputs
        ub = self.u_bias(users).reshape(users.shape)
        ib = self.i_bias(items).reshape(items.shape)

        uvec = self.u_embed(users)
        ivec = self.i_embed(items)

        # compute final score
        score = ub + ib + vecdot(uvec, ivec)

        # we're done
        return score
