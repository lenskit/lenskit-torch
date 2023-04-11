"""
Base classes for PyTorch for LensKit.
"""

import torch
from lenskit.algorithms import Algorithm

class LKTorchBase(Algorithm):
    """
    Base class for PyTorch algorithms, supporting basic infrastructure, serialization, etc.

    Attributes:
        _model:
            The algorithm's scoring model as a PyTorch module.
    """

    _configured_device = None
    _current_device = None
    _model: torch.nn.Module = None

    def __init__(self, *, device):
        self._configured_device = device

    def _default_device(self):
        "Get the device to use for this model."
        if self._current_device is not None:
            return self._current_device
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'gpu'

    def to(self, device):
        "Move the model to a different device."
        if self._model is not None:
            self._model.to(device)
        self._current_device = device
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_model' in state:
            del state['_model']
            state['_model_weights_'] = self._model.state_dict()
        if '_current_device' in state:
            # we always go back to CPU in pickling
            del state['_current_device']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_model_weights_' in state:
            self._prepare_model()
            self._model.load_state_dict(self._model_weights_)
            # set the model in evaluation mode (not training)
            self._model.eval()
            del self._model_weights_
