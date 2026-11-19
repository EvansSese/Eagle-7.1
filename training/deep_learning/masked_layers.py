"""Keras layers for sequence models with explicit timestep masks."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Eagle")
class MaskAwareGRU(tf.keras.layers.GRU):
    """Consume padded-timestep masks without leaking them past vector output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None


@tf.keras.utils.register_keras_serializable(package="Eagle")
class MaskAwareLSTM(tf.keras.layers.LSTM):
    """Consume padded-timestep masks without leaking them past vector output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None
