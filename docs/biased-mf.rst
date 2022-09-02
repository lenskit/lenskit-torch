Biased MF
=========

These models implement the standard biased matrix factorization model, like
:py:class:`lenskit.algorithms.als.BiasedMF`, but learn the model parameters
using TensorFlow's gradient descent instead of the alternating least squares
algorithm.  There are two implementations:

* :py:class:`lenskit_tf.BiasedMF` learns a matrix factorization to predict the
  residuals of :py:class:`lenskit.algorithms.bias.Bias`.
* :py:class:`lenskit_tf.IntegratedBiasMF` uses TensorFlow to learn the entire
  model, including both biases and embeddings.

Bias-Based
~~~~~~~~~~

.. autoclass:: lenskit_tf.BiasedMF

Fully Integrated
~~~~~~~~~~~~~~~~

.. autoclass:: lenskit_tf.IntegratedBiasMF
