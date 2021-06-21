
"""Keras-based TransformerEncoder block layer."""

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="keras_nlp")
class FourierMixingSublayer(tf.keras.layers.Layer):
    """FourierMixingSublayer layer.

    This layer implements the fourier mixing sublayer
    from "FNet: Mixing Tokens with Fourier Transforms".
    (https://arxiv.org/pdf/2105.03824v1.pdf)

    References:
        [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/pdf/2105.03824v1.pdf)
    """
    def __init__(self, **kwargs):
        super(FourierMixingSublayer, self).__init__()

    def call(self, query, **kwargs):
        casted_inputs = tf.cast(query, tf.complex64)
        
        fourier_output = tf.signal.fft2d(casted_inputs)
        return tf.math.real(fourier_output)