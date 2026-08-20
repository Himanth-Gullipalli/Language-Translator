"""
model.py  (sentiment)
──────────────────────
Bidirectional-LSTM sentiment classification model.

Architecture:
    Embedding  →  SpatialDropout1D
    →  BiLSTM (return_sequences)  →  Dropout
    →  BiLSTM (return_sequences)
    →  GlobalMaxPooling1D
    →  Dense(relu)  →  Dropout
    →  Dense(softmax)
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    SpatialDropout1D,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    SENT_VOCAB_SIZE,
    SENT_EMBEDDING_DIM,
    SENT_LSTM_UNITS,
    SENT_DROPOUT_RATE,
    SENT_MAX_LEN,
    SENTIMENT_MODEL_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_model(
    vocab_size:    int   = SENT_VOCAB_SIZE,
    embedding_dim: int   = SENT_EMBEDDING_DIM,
    lstm_units:    int   = SENT_LSTM_UNITS,
    dropout_rate:  float = SENT_DROPOUT_RATE,
    num_classes:   int   = 3,
    max_len:       int   = SENT_MAX_LEN,
) -> tf.keras.Model:
    """
    Build and compile the Bidirectional LSTM sentiment classifier.

    Args:
        vocab_size:    Source vocabulary size for the Embedding layer.
        embedding_dim: Embedding vector dimension.
        lstm_units:    LSTM units per direction in the first BiLSTM layer.
        dropout_rate:  Dropout probability (used for Spatial, LSTM, and Dense dropouts).
        num_classes:   Number of output classes (default 3: Negative / Neutral / Positive).
        max_len:       Input sequence length (informational; not used inside Sequential).

    Returns:
        Compiled tf.keras Sequential model.
    """
    model = Sequential(
        [
            # ── Embedding ──────────────────────────────────────────────────
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                mask_zero=True,
                name="embedding",
            ),
            SpatialDropout1D(dropout_rate, name="spatial_dropout"),

            # ── First Bidirectional LSTM ───────────────────────────────────
            Bidirectional(
                LSTM(
                    lstm_units,
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=0.1,
                ),
                name="bilstm_1",
            ),
            Dropout(dropout_rate, name="dropout_1"),

            # ── Second Bidirectional LSTM (smaller) ───────────────────────
            Bidirectional(
                LSTM(
                    lstm_units // 2,
                    return_sequences=True,
                    dropout=dropout_rate,
                ),
                name="bilstm_2",
            ),

            # ── Global max pooling ─────────────────────────────────────────
            GlobalMaxPooling1D(name="global_max_pool"),

            # ── Dense head ────────────────────────────────────────────────
            Dense(64, activation="relu", name="dense_1"),
            Dropout(dropout_rate, name="dropout_2"),

            # ── Output ────────────────────────────────────────────────────
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name="sentiment_bilstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_save_path: str = SENTIMENT_MODEL_PATH) -> list:
    """
    Standard training callbacks:
      • EarlyStopping    — stops when val_loss stops improving (patience 3).
      • ReduceLROnPlateau — halves LR when val_loss plateaus (patience 2).
      • ModelCheckpoint  — saves the best model (by val_accuracy) to disk.

    Args:
        model_save_path: Path where the best model is saved (.keras format).

    Returns:
        List of tf.keras.callbacks.Callback instances.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
