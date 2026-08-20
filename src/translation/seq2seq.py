"""
seq2seq.py  (translation)
──────────────────────────
Seq2Seq training wrapper.

Combines TranslationEncoder + TranslationDecoder into a single class that:
  • Runs teacher-forced training steps via a custom training loop
  • Computes masked sparse-categorical-crossentropy loss
  • Applies gradient clipping to stabilise training
  • Saves / loads encoder and decoder weights separately
"""

import os
import sys
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    TRANS_EMBEDDING_DIM,
    TRANS_UNITS,
    get_encoder_weights_path,
    get_decoder_weights_path,
)
from src.translation.encoder import TranslationEncoder
from src.translation.decoder import TranslationDecoder


class Seq2SeqTrainer:
    """
    Trainer for the Encoder–Decoder Seq2Seq model.

    Usage
    -----
    trainer = Seq2SeqTrainer(eng_vocab, tgt_vocab)
    for epoch in range(EPOCHS):
        for eng_batch, tgt_batch in dataset:
            loss = trainer.train_step(eng_batch, tgt_batch, start_id)
    trainer.save_weights("french")
    """

    def __init__(
        self,
        eng_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim:  int   = TRANS_EMBEDDING_DIM,
        units:          int   = TRANS_UNITS,
        learning_rate:  float = 1e-3,
    ):
        self.encoder = TranslationEncoder(
            vocab_size=eng_vocab_size,
            embedding_dim=embedding_dim,
            enc_units=units,
        )
        self.decoder = TranslationDecoder(
            vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            dec_units=units,
            attention_units=units,
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=5.0
        )
        # from_logits=True because decoder outputs raw logits
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────

    def _masked_loss(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """
        Compute mean loss while masking padding positions (token id == 0).

        Args:
            real: True target token ids, shape (batch,).
            pred: Predicted logits,       shape (batch, vocab_size).

        Returns:
            Scalar mean loss over non-padding positions.
        """
        mask  = tf.math.logical_not(tf.math.equal(real, 0))
        loss  = self._loss_fn(real, pred)          # (batch,)
        mask  = tf.cast(mask, dtype=loss.dtype)
        denom = tf.reduce_sum(mask)
        return tf.reduce_sum(loss * mask) / (denom + 1e-8)

    # ──────────────────────────────────────────────────────────────────────
    # Training Step
    # ──────────────────────────────────────────────────────────────────────

    @tf.function
    def train_step(
        self,
        eng_batch:          tf.Tensor,
        tgt_batch:          tf.Tensor,
        tgt_start_token_id: int,
    ) -> tf.Tensor:
        """
        One training step with teacher forcing.

        Args:
            eng_batch:          Source sequences, shape (batch, src_len).
            tgt_batch:          Target sequences (incl. <start> and <end>),
                                shape (batch, tgt_len).
            tgt_start_token_id: Integer ID of the <start> token.

        Returns:
            Scalar batch loss (averaged over time steps).
        """
        batch_size = tf.shape(eng_batch)[0]
        tgt_len    = tgt_batch.shape[1]
        total_loss = 0.0

        with tf.GradientTape() as tape:
            # ── Encode ──────────────────────────────────────────────────
            enc_output, state_h, state_c = self.encoder(eng_batch, training=True)
            dec_hidden = state_h  # initialise decoder with encoder's final state

            # Decoder first input: <start> token for every example in batch
            dec_input = tf.expand_dims(
                tf.fill([batch_size], tgt_start_token_id), axis=1
            )  # (batch, 1)

            # ── Teacher-forced decode ────────────────────────────────────
            for t in range(1, tgt_len):
                logits, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output, training=True
                )
                # logits: (batch, 1, vocab_size)
                total_loss += self._masked_loss(
                    tgt_batch[:, t],        # (batch,)
                    logits[:, 0, :],        # (batch, vocab_size)
                )
                # Next input is the true token (teacher forcing)
                dec_input = tf.expand_dims(tgt_batch[:, t], axis=1)

            batch_loss = total_loss / tf.cast(tgt_len, tf.float32)

        # ── Gradients ───────────────────────────────────────────────────
        variables  = (self.encoder.trainable_variables
                      + self.decoder.trainable_variables)
        gradients  = tape.gradient(batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # ──────────────────────────────────────────────────────────────────────
    # Weight I/O
    # ──────────────────────────────────────────────────────────────────────

    def save_weights(self, language: str) -> None:
        """
        Save encoder and decoder weights to the models/translation/ directory.

        Args:
            language: Target language name (used to construct filenames).
        """
        enc_path = get_encoder_weights_path(language)
        dec_path = get_decoder_weights_path(language)
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)

        self.encoder.save_weights(enc_path)
        self.decoder.save_weights(dec_path)

        print(f"[INFO] Encoder weights saved  →  {enc_path}")
        print(f"[INFO] Decoder weights saved  →  {dec_path}")

    def load_weights(self, language: str) -> None:
        """
        Load encoder and decoder weights from the models/translation/ directory.

        Args:
            language: Target language name.
        """
        enc_path = get_encoder_weights_path(language)
        dec_path = get_decoder_weights_path(language)

        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)

        print(f"[INFO] Encoder weights loaded  ←  {enc_path}")
        print(f"[INFO] Decoder weights loaded  ←  {dec_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Dataset Utility
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_tf_dataset(
        eng_seqs: np.ndarray,
        tgt_seqs: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        buffer_size: int = 10_000,
    ) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset from preprocessed numpy arrays.

        Args:
            eng_seqs:    Padded English sequences, shape (N, src_len).
            tgt_seqs:    Padded target sequences, shape (N, tgt_len).
            batch_size:  Batch size.
            shuffle:     Whether to shuffle the dataset.
            buffer_size: Shuffle buffer size.

        Returns:
            tf.data.Dataset yielding (eng_batch, tgt_batch) tuples.
        """
        dataset = tf.data.Dataset.from_tensor_slices((eng_seqs, tgt_seqs))
        if shuffle:
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE
        )
        return dataset
