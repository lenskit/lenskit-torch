import logging
import pickle
import binpickle
import seedbank
from pickletools import dis as pickle_dis

from pytest import mark, approx

import lenskit.util.test as lktu
from lkbuild.data import ml_small

import pandas as pd
import numpy as np

from lktorch.biasedmf import TorchBiasedMF

_log = logging.getLogger(__name__)


@lktu.wantjit
@mark.slow
def test_biasmf_train_large():
    algo = TorchBiasedMF(20)
    ratings = ml_small.ratings
    algo.fit(ratings)

    rng = seedbank.numpy_rng()

    # let's get some predictions
    for user, item in zip(rng.choice(ratings['user'].unique(), 100), rng.choice(ratings['item'].unique(), 100)):
        pred = algo.predict_for_user(user, np.array([item]))
        assert np.isfinite(pred.loc[item])
        assert pred.loc[item] >= -10
        assert pred.loc[item] <= 15


@mark.skipif(not binpickle, reason='binpickle not available')
def test_biasmf_binpickle(tmp_path):
    "Test saving TorchBiasedMF with BinPickle"

    original = TorchBiasedMF(20, epochs=2)
    ratings = ml_small.ratings
    original.fit(ratings)

    file = tmp_path / 'biasmf.bpk'
    binpickle.dump(original, file)

    with binpickle.BinPickleFile(file) as bpf:
        # the pickle data should be small
        _log.info('serialized to %d pickle bytes', bpf.entries[-1].dec_length)
        pickle_dis(bpf._read_buffer(bpf.entries[-1]))
        assert bpf.entries[-1].dec_length < 2048

        algo = bpf.load()

        orig_state = original._model.state_dict()
        load_state = algo._model.state_dict()
        for k in orig_state:
            _log.info('checking %s', k)
            assert np.allclose(orig_state[k], load_state[k])

        # make sure it still works
        preds = algo.predict_for_user(10, np.arange(0, 50, dtype='i8'))
        assert len(preds) == 50
