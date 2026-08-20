"""
encoder.py  (translation)
──────────────────────────
Encoder for the Seq2Seq NMT model.

Architecture:
    Embedding  →  Bidirectional LSTM  →  Dense projections

The encoder reads the full English source sentence and returns:
  • encoder_output — all hidden states (used by Bahdanau attention)
  • state_h        — projected final hidden state (decoder init)
  • state_c        — projected final cell  state  (decoder init)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout
)


class TranslationEncoder(tf.keras.Model):
    """
    Bidirectional-LSTM encoder.

    Parameters
    ----------
    vocab_size    : int  — Source (English) vocabulary size.
    embedding_dim : int  — Embedding vector dimension.
    enc_units     : int  — LSTM units *per direction* (total hidden is 2×enc_units).
    dropout_rate  : float — Dropout on embeddings and recurrent weights.
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        enc_units:     int,
        dropout_rate:  float = 0.2,
        **kwargs,
    ):
        super(TranslationEncoder, self).__init__(**kwargs)
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units     = enc_units
        self.dropout_rate  = dropout_rate

        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,          # mask padding tokens
            name="enc_embedding",
        )
        self.dropout = Dropout(dropout_rate, name="enc_emb_dropout")

        # Bidirectional LSTM returns sequences + both forward & backward states
        self.bilstm = Bidirectional(
            LSTM(
                enc_units,
                return_sequences=True,
                return_state=True,
                dropout=dropout_rate,
                recurrent_dropout=0.1,
            ),
            name="enc_bilstm",
        )

        # Project concatenated fwd+bwd states → enc_units (decoder size)
        self.fc_h = Dense(enc_units, activation="tanh", name="enc_fc_h")
        self.fc_c = Dense(enc_units, activation="tanh", name="enc_fc_c")

    def call(self, x: tf.Tensor, training: bool = False):
        """
        Forward pass.

        Args:
            x:        Source token indices, shape (batch, src_len).
            training: Boolean flag for dropout.

        Returns:
            encoder_output : All hidden states, shape (batch, src_len, 2*enc_units).
            state_h        : Projected final hidden state, shape (batch, enc_units).
            state_c        : Projected final cell  state, shape (batch, enc_units).
        """
        # Embedding + dropout
        embedded = self.embedding(x)                            # (batch, src_len, emb_dim)
        embedded = self.dropout(embedded, training=training)

        # BiLSTM → outputs + 4 state tensors (fwd_h, fwd_c, bwd_h, bwd_c)
        enc_output, fwd_h, fwd_c, bwd_h, bwd_c = self.bilstm(
            embedded, training=training
        )   # enc_output: (batch, src_len, 2*enc_units)

        # Merge forward and backward terminal states
        state_h = self.fc_h(tf.concat([fwd_h, bwd_h], axis=-1))  # (batch, enc_units)
        state_c = self.fc_c(tf.concat([fwd_c, bwd_c], axis=-1))  # (batch, enc_units)

        return enc_output, state_h, state_c

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size":    self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "enc_units":     self.enc_units,
            "dropout_rate":  self.dropout_rate,
        })
        return cfg
