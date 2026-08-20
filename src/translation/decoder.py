"""
decoder.py  (translation)
──────────────────────────
GRU-based decoder with Bahdanau attention for the Seq2Seq NMT model.

At each time step the decoder:
  1. Embeds the previous (or true) target token.
  2. Computes an attention context vector over encoder outputs.
  3. Concatenates context + embedding → GRU input.
  4. Projects GRU output to target vocabulary logits.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.translation.attention import BahdanauAttention


class TranslationDecoder(tf.keras.Model):
    """
    GRU decoder with Bahdanau attention.

    Parameters
    ----------
    vocab_size       : int   — Target language vocabulary size.
    embedding_dim    : int   — Embedding vector dimension.
    dec_units        : int   — GRU hidden units.
    attention_units  : int   — Units inside the attention scoring layers.
    dropout_rate     : float — Dropout probability.
    """

    def __init__(
        self,
        vocab_size:      int,
        embedding_dim:   int,
        dec_units:       int,
        attention_units: int,
        dropout_rate:    float = 0.2,
        **kwargs,
    ):
        super(TranslationDecoder, self).__init__(**kwargs)
        self.vocab_size      = vocab_size
        self.embedding_dim   = embedding_dim
        self.dec_units       = dec_units
        self.attention_units = attention_units
        self.dropout_rate    = dropout_rate

        self.embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=False,   # attention handles masking already
            name="dec_embedding",
        )

        self.attention = BahdanauAttention(attention_units, name="bahdanau_attention")

        self.gru = GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=0.1,
            name="dec_gru",
        )

        self.dropout = Dropout(dropout_rate, name="dec_dropout")
        # Output logits — softmax is applied in the loss (from_logits=True)
        self.fc_out  = Dense(vocab_size, name="dec_output_fc")

    def call(
        self,
        x:              tf.Tensor,
        hidden:         tf.Tensor,
        encoder_output: tf.Tensor,
        training:       bool = False,
    ):
        """
        Single decoder time step.

        Args:
            x:              Current input token, shape (batch, 1).
            hidden:         Previous GRU hidden state, shape (batch, dec_units).
            encoder_output: All encoder states, shape (batch, src_len, 2*enc_units).
            training:       Boolean flag for dropout.

        Returns:
            logits:           Token prediction logits, shape (batch, 1, vocab_size).
            new_hidden:       Updated GRU hidden state, shape (batch, dec_units).
            attention_weights: Alignment weights, shape (batch, src_len, 1).
        """
        # ── Attention ───────────────────────────────────────────────────────
        context_vector, attention_weights = self.attention(encoder_output, hidden)
        # context_vector: (batch, 2*enc_units)

        # ── Embedding ────────────────────────────────────────────────────────
        embedded = self.embedding(x)          # (batch, 1, emb_dim)

        # ── Concatenate context + embedding ──────────────────────────────────
        context_exp = tf.expand_dims(context_vector, axis=1)  # (batch, 1, 2*enc_units)
        gru_input   = tf.concat([context_exp, embedded], axis=-1)
        # shape: (batch, 1, 2*enc_units + emb_dim)

        # ── GRU ──────────────────────────────────────────────────────────────
        gru_out, new_hidden = self.gru(
            gru_input, initial_state=hidden, training=training
        )
        # gru_out:    (batch, 1, dec_units)
        # new_hidden: (batch, dec_units)

        gru_out = self.dropout(gru_out, training=training)
        logits  = self.fc_out(gru_out)        # (batch, 1, vocab_size)

        return logits, new_hidden, attention_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "vocab_size":      self.vocab_size,
            "embedding_dim":   self.embedding_dim,
            "dec_units":       self.dec_units,
            "attention_units": self.attention_units,
            "dropout_rate":    self.dropout_rate,
        })
        return cfg
