"""
preprocessing.py  (sentiment)
──────────────────────────────
Text cleaning, tokenisation, label encoding, and dataset preparation
for the sentiment analysis model.
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
# Download stopwords on first use (silent if already present)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

_STOPWORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean raw text for sentiment classification.

    Steps:
        1. Lowercase + strip
        2. Remove HTML tags
        3. Remove URLs
        4. Replace non-alphabetic characters with spaces
        5. Optionally remove English stopwords
        6. Collapse repeated whitespace

    Args:
        text:             Raw input string.
        remove_stopwords: Whether to strip common English stopwords.

    Returns:
        Cleaned string containing only lowercase letters and single spaces.
    """
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>",      " ", text)   # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s]",     " ", text)   # non-alpha
    text = re.sub(r"\s+",          " ", text).strip()

    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in _STOPWORDS)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sentiment_data(
    csv_path:  str,
    text_col:  str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Load a sentiment CSV and return a cleaned DataFrame.

    Expected CSV columns: text_col (raw text), label_col (sentiment label string).

    Args:
        csv_path:  Absolute path to the CSV file.
        text_col:  Column name for the input text.
        label_col: Column name for the sentiment label.

    Returns:
        DataFrame with exactly two columns: 'text' (cleaned) and 'label'.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sentiment dataset not found: {csv_path}\n"
            "  → Place 'sentiment.csv' in 'datasets/sentiment/'"
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not in {csv_path}.\n"
            f"  Available: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df.columns = ["text", "label"]
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    print(f"[INFO] Loaded {len(df):,} sentiment samples.")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Label Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(labels: list) -> tuple:
    """
    Encode string labels to integer class indices using sklearn LabelEncoder
    (alphabetical ordering: negative → 0, neutral → 1, positive → 2).

    Args:
        labels: List of string sentiment labels.

    Returns:
        Tuple of (encoded_array: np.ndarray, fitted_LabelEncoder).
    """
    le      = LabelEncoder()
    encoded = le.fit_transform(labels)
    print(f"[INFO] Label classes (alphabetical): {list(le.classes_)}")
    return encoded, le


def save_label_encoder(le: LabelEncoder, path: str) -> None:
    """Serialize a LabelEncoder to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(le, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] LabelEncoder saved  →  {path}")


def load_label_encoder(path: str) -> LabelEncoder:
    """Load a LabelEncoder from a pickle file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation & Padding
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(texts: list, vocab_size: int) -> Tokenizer:
    """Fit and return a Keras Tokenizer."""
    tok = Tokenizer(num_words=vocab_size, oov_token="<oov>")
    tok.fit_on_texts(texts)
    print(f"[INFO] Sentiment tokenizer fitted — vocab: {len(tok.word_index):,} "
          f"(capped at {vocab_size:,})")
    return tok


def texts_to_padded(
    tokenizer:  Tokenizer,
    texts:      list,
    maxlen:     int,
    padding:    str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Convert texts to padded integer sequences."""
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding=padding, truncating=truncating)


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """Serialize tokenizer to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Sentiment tokenizer saved  →  {path}")


def load_tokenizer(path: str) -> Tokenizer:
    """Load tokenizer from pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_sentiment_dataset(
    csv_path:            str,
    vocab_size:          int,
    maxlen:              int,
    test_size:           float = 0.2,
    tokenizer_save_path: str   = None,
    label_enc_save_path: str   = None,
) -> dict:
    """
    End-to-end preprocessing for sentiment analysis:

    1. Load CSV + clean text
    2. Encode string labels → integers
    3. Stratified train / test split
    4. Fit tokeniser on training data only
    5. Pad sequences

    Args:
        csv_path:            Path to sentiment.csv.
        vocab_size:          Maximum vocabulary size.
        maxlen:              Maximum token sequence length.
        test_size:           Fraction held out for testing.
        tokenizer_save_path: If given, save the fitted tokeniser here.
        label_enc_save_path: If given, save the LabelEncoder here.

    Returns:
        dict with keys:
            X_train, X_test  — padded np.ndarray sequences
            y_train, y_test  — integer label arrays
            tokenizer        — fitted Keras Tokenizer
            label_encoder    — fitted sklearn LabelEncoder
            num_classes      — int, number of unique sentiment classes
    """
    df = load_sentiment_data(csv_path)
    labels_encoded, label_encoder = encode_labels(df["label"].tolist())

    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"].tolist(),
        labels_encoded,
        test_size=test_size,
        random_state=42,
        stratify=labels_encoded,
    )
    print(f"[INFO] Split → Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

    tokenizer = build_tokenizer(X_train_raw, vocab_size)
    X_train   = texts_to_padded(tokenizer, X_train_raw, maxlen)
    X_test    = texts_to_padded(tokenizer, X_test_raw,  maxlen)

    if tokenizer_save_path:
        save_tokenizer(tokenizer, tokenizer_save_path)
    if label_enc_save_path:
        save_label_encoder(label_encoder, label_enc_save_path)

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "tokenizer":     tokenizer,
        "label_encoder": label_encoder,
        "num_classes":   len(label_encoder.classes_),
    }
