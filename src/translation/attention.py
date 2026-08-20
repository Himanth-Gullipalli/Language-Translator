"""
attention.py  (translation)
────────────────────────────
Bahdanau (additive) attention mechanism for use inside the Seq2Seq decoder.

Reference:
    Bahdanau, Cho & Bengio (2015) — "Neural Machine Translation by
    Jointly Learning to Align and Translate"
    https://arxiv.org/abs/1409.0473
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Additive (Bahdanau) attention layer.

    At each decoder time step the layer computes:
        score(h_t, H_s) = V · tanh( W1·H_s  +  W2·h_t )
        α               = softmax(score)
        context         = Σ α · H_s

    where h_t is the current decoder hidden state and H_s is the full
    sequence of encoder hidden states.

    Parameters
    ----------
    units : int
        Number of units inside the scoring dense layers (W1, W2, V).
    """

    def __init__(self, units: int, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units, use_bias=False, name="attn_W1")   # encoder projection
        self.W2 = Dense(units, use_bias=False, name="attn_W2")   # decoder projection
        self.V  = Dense(1,     use_bias=False, name="attn_V")    # scalar score

    def call(
        self,
        encoder_output: tf.Tensor,
        decoder_hidden: tf.Tensor,
    ):
        """
        Compute context vector and attention weights.

        Args:
            encoder_output: All encoder hidden states,
                            shape (batch, src_len, enc_units).
            decoder_hidden: Decoder hidden state at current step,
                            shape (batch, dec_units).

        Returns:
            context_vector:   Weighted sum of encoder outputs,
                              shape (batch, enc_units).
            attention_weights: Normalised alignment scores,
                               shape (batch, src_len, 1).
        """
        # Expand decoder hidden state for broadcasting over src_len
        # (batch, dec_units)  →  (batch, 1, dec_units)
        dec_hidden_exp = tf.expand_dims(decoder_hidden, axis=1)

        # Score computation: (batch, src_len, 1)
        score = self.V(
            tf.nn.tanh(
                self.W1(encoder_output)   # (batch, src_len, units)
                + self.W2(dec_hidden_exp) # (batch, 1,       units)  — broadcast
            )
        )

        # Normalise across src_len → (batch, src_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Weighted sum → (batch, enc_units)
        context_vector = tf.reduce_sum(
            attention_weights * encoder_output, axis=1
        )

        return context_vector, attention_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
