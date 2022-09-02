"""
PyTorch-based algorithms.
"""

import logging

from . import _pt

__version__ = '0.14.0'
__version__ = _pt.gitify_version(__version__)

_log = logging.getLogger(__name__)
