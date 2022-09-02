TensorFlow for LensKit
======================

.. module:: lenskit_tf

.. _TensorFlow: https://tensorflow.org

This package provides algorithm implementations, particularly matrix
factorization, using TensorFlow_.  These algorithms serve two purposes:

* Provide classic algorithms ready to use for recommendation or as
  baselines for new techniques.
* Demonstrate how to connect TensorFlow to LensKit for use in your own
  experiments.

To install::

    pip install lenskit-tf

Or (preferred, once published)::

    conda install -c conda-forge lenskit-tf

.. warning::
    These implementations are not yet battle-tested --- they are here
    primarily for demonstration purposes at this time.

.. toctree::
    :caption: Contents

    biased-mf
    bpr
