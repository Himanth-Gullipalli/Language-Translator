"""
preprocessing.py  (translation)
────────────────────────────────
Text cleaning, tokenisation, padding, and tokeniser I/O for the
neural machine translation pipeline.
"""

import re
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Allow running from any working directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import START_TOKEN, END_TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise and clean a single sentence.

    Steps:
        1. Lowercase + strip
        2. Keep letters (Latin + Devanagari), digits, and basic spaces
        3. Collapse repeated whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned, lowercased string.
    """
    text = str(text).lower().strip()
    # Keep Unicode letters (covers French, German, Spanish accents, Hindi Devanagari)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_special_tokens(sentences: list) -> list:
    """
    Wrap each target sentence with <start> and <end> tokens.
    These are required so the decoder knows when to start and stop.

    Args:
        sentences: List of cleaned target-language strings.

    Returns:
        List of strings, each wrapped as  "<start> ... <end>".
    """
    return [f"{START_TOKEN} {s} {END_TOKEN}" for s in sentences]


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(sentences: list, vocab_size: int) -> Tokenizer:
    """
    Fit a Keras Tokenizer on a corpus of sentences.

    Args:
        sentences:  List of text strings to fit on.
        vocab_size: Maximum vocabulary size (most-frequent words kept).

    Returns:
        Fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<oov>",
        filters="",          # Don't strip special tokens like <start>
    )
    tokenizer.fit_on_texts(sentences)
    print(f"[INFO] Tokenizer fitted — vocab size: {len(tokenizer.word_index):,} "
          f"(capped at {vocab_size:,})")
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    sentences: list,
    maxlen: int,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """
    Convert a list of sentences to zero-padded integer sequences.

    Args:
        tokenizer:  Fitted Keras Tokenizer.
        sentences:  List of text strings to encode.
        maxlen:     Desired fixed sequence length.
        padding:    'pre' or 'post' — where to add padding zeros.
        truncating: 'pre' or 'post' — where to truncate long sequences.

    Returns:
        NumPy array of shape (len(sentences), maxlen).
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    padded    = pad_sequences(
        sequences, maxlen=maxlen, padding=padding, truncating=truncating
    )
    return padded


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """
    Serialise a fitted Tokenizer to a pickle file.

    Args:
        tokenizer: Fitted Keras Tokenizer.
        path:      Absolute path for the output .pkl file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Tokenizer saved  →  {path}")


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a Tokenizer from a pickle file.

    Args:
        path: Absolute path to a .pkl tokenizer file.

    Returns:
        Loaded Keras Tokenizer instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    with open(path, "rb") as fh:
        tokenizer = pickle.load(fh)
    print(f"[INFO] Tokenizer loaded  ←  {path}")
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_inference(
    sentence: str,
    tokenizer: Tokenizer,
    maxlen: int,
) -> np.ndarray:
    """
    Clean and encode a single sentence for model inference.

    Args:
        sentence:  Raw English input text.
        tokenizer: Fitted source-language Tokenizer.
        maxlen:    Maximum sequence length expected by the encoder.

    Returns:
        NumPy array of shape (1, maxlen) ready to feed to the encoder.
    """
    cleaned  = clean_text(sentence)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=maxlen, padding="post")
    return padded
