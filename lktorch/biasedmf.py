"""
Biased matrix factorization for explicit-feedback data.
"""

import logging

from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


from lenskit.algorithms import Predictor
from lenskit import util

from lktorch.util import torch_dot
from lktorch.data.batch import BatchSampler
from lktorch.data.ratings import RatingData

from tqdm.auto import tqdm

_log = logging.getLogger(__name__)


class MFNet(nn.Module):
    """
    Torch module that defines the matrix factorization model.

    Args:
        n_users(int): the number of users
        n_items(int): the number of items
        n_feats(int): the embedding dimension
    """

    def __init__(self, n_users, n_items, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.n_users = n_users
        self.n_items = n_items

        # global bias term
        self.g_bias = nn.Parameter(torch.as_tensor(0.0))

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

    def forward(self, user, item):
        """
        Compute the forward pass for a batch.

        Args:
            user(torch.Tensor):
                The user IDs to score.
            item(torch.Tensor):
                The item IDs to score.
        """
        # look up biases and embeddings
        # reshape converts arrays of shape (B, 1) to shape B
        ub = self.u_bias(user).reshape(-1)
        ib = self.i_bias(item).reshape(-1)

        uvec = self.u_embed(user)
        ivec = self.i_embed(item)

        # compute the inner score
        score = self.g_bias + ub + ib + torch_dot(uvec, ivec)

        # we're done
        assert score.shape == user.shape
        return score


class TorchBiasedMF(Predictor):
    """
    Implementation of MF in Torch, with sigmoid activation.
    """

    _device = None
    _train_dev = None

    def __init__(self, n_features, *, batch_size=8*1024, epochs=5,
                 lrate=0.001, weight_decay=0.01, device=None, rng_spec=None):
        """
        Initialize the Torch MF predictor.

        Args:
            n_features(int):
                The number of latent features (embedding size).
            batch_size(int):
                The batch size for training.  Since this model is relatively simple,
                large batch sizes work well.
            lrate(float):
                The learning rate for :class:`~torch.optim.AdamW`.
            weight_decay(float):
                The regularization weight for :class:`~torch.optim.AdamW`.
            epochs(int):
                The number of training epochs to run.
            device(str):
                The device to use.
            rng_spec:
                The random number specification (for data, not for Torch initialization).
        """

        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.lrate = lrate
        self.weight_decay = weight_decay
        self.rng_spec = rng_spec

        self._device = device

    def fit(self, ratings, **kwargs):
        # run the iterations
        timer = util.Stopwatch()

        _log.info('[%s] preparing input data set', timer)
        data = RatingData.from_ratings(ratings)
        self.user_index_ = data.users
        self.item_index_ = data.items

        dev = self._device
        if dev is None:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._prepare_model(dev)

        # initialize model to global mean
        self._model.g_bias.data = torch.as_tensor(np.mean(data.r_values))

        # now __model has the trainable model
        batches = BatchSampler(data, self.batch_size, self.rng_spec)
        for epoch in tqdm(range(self.epochs), 'epoch', leave=False):
            _log.info('[%s] beginning epoch %d', timer, epoch + 1)

            batches.shuffle()
            self._fit_iter(batches)

            unorm = torch.linalg.norm(self._model.u_embed.weight.data).item()
            inorm = torch.linalg.norm(self._model.i_embed.weight.data).item()
            _log.info('[%s] epoch %d finished (|P|=%.3f, |Q|=%.3f, b=%.3f)',
                      timer, epoch + 1, unorm, inorm, self._model.g_bias.data.item())

        _log.info('finished training')
        self._finalize()
        self._cleanup()
        return self

    def _prepare_model(self, train_dev=None):
        model = MFNet(len(self.user_index_), len(self.item_index_), self.n_features)
        self._model = model
        if train_dev:
            _log.info('preparing to train on %s', train_dev)
            self._train_dev = train_dev
            # move device to model
            self._model = model.to(train_dev)
            # set up training features
            self._loss = nn.MSELoss()
            self._opt = AdamW(self._model.parameters(), self.lrate, weight_decay=self.weight_decay)

    def _finalize(self):
        "Finalize model training, moving back to the CPU"
        self._model = self._model.to('cpu')
        del self._train_dev

    def _cleanup(self):
        "Clean up data not needed after training"
        del self._loss, self._opt

    def _fit_iter(self, batches):
        """
        Run one iteration of the recommender training.
        """

        loop = tqdm(range(batches.batch_count), leave=False)
        for i in loop:
            batch = batches.batch(i).to(self._train_dev)
            uv, iv, rv = batch

            # compute scores and loss
            pred = self._model(uv, iv)
            pred_loss = self._loss(pred, rv)

            # update model
            self._opt.zero_grad()
            pred_loss.backward()
            self._opt.step()

            loop.set_postfix_str('loss: {:.3f}'.format(pred_loss.item()))

            _log.debug('batch %d has loss %s', i, pred_loss.item())

        loop.clear()

    def predict_for_user(self, user, items, ratings=None):
        """
        Generate item scores for a user.

        This needs to do two things:

        1. Look up the user's embeddings
        2. Score the items using them

        Note that user and items are both user and item IDs, not positions, so we need
        to convert them.
        """

        # convert user and items into rows and columns
        u_row = self.user_index_.get_loc(user)
        i_cols = self.item_index_.get_indexer(items)
        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable = items[i_cols >= 0]
        i_cols = i_cols[i_cols >= 0]

        u_tensor = torch.from_numpy(np.repeat(u_row, len(i_cols)))
        i_tensor = torch.from_numpy(i_cols)
        if self._train_dev:
            u_tensor = u_tensor.to(self._train_dev)
            i_tensor = i_tensor.to(self._train_dev)

        # get scores
        with torch.no_grad():
            scores = self._model(u_tensor, i_tensor).to('cpu')

        # and we can finally put in a series to return
        results = pd.Series(scores, index=scorable)
        return results.reindex(items)  # fill in missing values with nan

    def __str__(self):
        return 'TorchBiasedMF(features={}, reg={})'.format(self.n_features, self.reg)

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_model' in state:
            del state['_model']
            weights = self._model.state_dict()
            weights = OrderedDict((k, t.numpy()) for (k, t) in weights.items())
            state['_model_weights_'] = weights

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_model_weights_' in state:
            self._prepare_model()
            weights = OrderedDict((k, torch.from_numpy(a)) for (k, a) in self._model_weights_.items())
            self._model.load_state_dict(weights)
            self._model.eval()
            del self._model_weights_
