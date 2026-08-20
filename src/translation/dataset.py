"""
dataset.py  (translation)
──────────────────────────
Loads raw CSV data, runs the full preprocessing pipeline, and saves
preprocessed artifacts + tokenisers to disk.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    TRANSLATION_DATASET_DIR,
    TRANSLATION_ARTIFACT_DIR,
    LANGUAGE_CSV_MAP,
    LANGUAGE_COL_MAP,
    TRANS_ENG_VOCAB_SIZE,
    TRANS_TARGET_VOCAB_SIZE,
    TRANS_MAX_ENG_LEN,
    TRANS_MAX_TARGET_LEN,
    get_eng_tokenizer_path,
    get_target_tokenizer_path,
    get_translation_artifact_path,
)
from src.common import ensure_dirs
from src.translation.preprocessing import (
    clean_text,
    add_special_tokens,
    build_tokenizer,
    texts_to_padded_sequences,
    save_tokenizer,
    load_tokenizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Raw Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(language: str, nrows: int = None) -> pd.DataFrame:
    """
    Load the raw translation CSV for a given language pair.

    Expected CSV columns: 'english',  '{language}'
    e.g. for French: 'english', 'french'

    Args:
        language: Target language key from SUPPORTED_LANGUAGES.
        nrows:    Optional row limit — useful for quick experiments.

    Returns:
        DataFrame containing only the required two columns, NaN rows dropped.

    Raises:
        FileNotFoundError: If the expected CSV file is missing.
        ValueError:        If required columns are absent from the CSV.
    """
    csv_path = os.path.join(TRANSLATION_DATASET_DIR, LANGUAGE_CSV_MAP[language])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"  → Place '{LANGUAGE_CSV_MAP[language]}' inside 'datasets/translation/'"
        )

    df = pd.read_csv(csv_path, nrows=nrows)

    required_cols = ["english", LANGUAGE_COL_MAP[language]]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in {csv_path}.\n"
            f"  Available columns: {list(df.columns)}"
        )

    df = df[required_cols].dropna().reset_index(drop=True)
    print(f"[INFO] Loaded {len(df):,} sentence pairs for "
          f"English → {language.capitalize()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_translation_dataset(
    language: str,
    test_size: float = 0.1,
    nrows: int = None,
) -> dict:
    """
    End-to-end preprocessing for one language pair:

    1. Load raw CSV
    2. Clean text (both sides)
    3. Add <start>/<end> tokens to the target side
    4. Split into train / test sets
    5. Build & fit tokenisers on training data only
    6. Convert to padded integer sequences
    7. Save tokenisers (.pkl) and padded arrays (artifacts .pkl)

    Args:
        language:  Target language (must be in SUPPORTED_LANGUAGES).
        test_size: Fraction of data to hold out for testing.
        nrows:     Optional row limit.

    Returns:
        dict with keys:
            eng_train, eng_test  — padded English sequences (np.ndarray)
            tgt_train, tgt_test  — padded target sequences (np.ndarray)
            eng_tokenizer        — fitted Keras Tokenizer (source)
            tgt_tokenizer        — fitted Keras Tokenizer (target)
    """
    ensure_dirs(TRANSLATION_ARTIFACT_DIR, TRANSLATION_DATASET_DIR)
    ensure_dirs(os.path.dirname(get_eng_tokenizer_path(language)))

    target_col = LANGUAGE_COL_MAP[language]

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_raw_data(language, nrows=nrows)

    df = df.sample(50000, random_state=42)
    df = df.reset_index(drop=True)
    # print(df.shape)
    # ── 2. Clean ──────────────────────────────────────────────────────────
    eng_sentences = [clean_text(s) for s in df["english"].tolist()]
    tgt_sentences = [clean_text(s) for s in df[target_col].tolist()]

    # ── 3. Special tokens on target ────────────────────────────────────────
    tgt_with_tokens = add_special_tokens(tgt_sentences)

    # ── 4. Train / Test split ─────────────────────────────────────────────
    (eng_train_raw, eng_test_raw,
     tgt_train_raw, tgt_test_raw) = train_test_split(
        eng_sentences, tgt_with_tokens,
        test_size=test_size, random_state=42,
    )
    print(f"[INFO] Split → Train: {len(eng_train_raw):,}  |  Test: {len(eng_test_raw):,}")

    # ── 5. Build tokenisers on train only ──────────────────────────────────
    eng_tokenizer = build_tokenizer(eng_train_raw, TRANS_ENG_VOCAB_SIZE)
    tgt_tokenizer = build_tokenizer(tgt_train_raw, TRANS_TARGET_VOCAB_SIZE)

    # ── 6. Pad sequences ──────────────────────────────────────────────────
    eng_train = texts_to_padded_sequences(eng_tokenizer, eng_train_raw, TRANS_MAX_ENG_LEN)
    eng_test  = texts_to_padded_sequences(eng_tokenizer, eng_test_raw,  TRANS_MAX_ENG_LEN)
    tgt_train = texts_to_padded_sequences(tgt_tokenizer, tgt_train_raw, TRANS_MAX_TARGET_LEN)
    tgt_test  = texts_to_padded_sequences(tgt_tokenizer, tgt_test_raw,  TRANS_MAX_TARGET_LEN)

    print(f"[INFO] Shapes → eng_train: {eng_train.shape}  tgt_train: {tgt_train.shape}")

    # ── 7. Save tokenisers ────────────────────────────────────────────────
    save_tokenizer(eng_tokenizer, get_eng_tokenizer_path(language))
    save_tokenizer(tgt_tokenizer, get_target_tokenizer_path(language))

    # ── 7. Save artifacts ─────────────────────────────────────────────────
    train_path = get_translation_artifact_path(language, "train")
    test_path  = get_translation_artifact_path(language, "test")

    with open(train_path, "wb") as fh:
        pickle.dump({"eng": eng_train, "tgt": tgt_train}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_path, "wb") as fh:
        pickle.dump({"eng": eng_test,  "tgt": tgt_test},  fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Artifacts saved → {train_path}")
    print(f"[INFO] Artifacts saved → {test_path}")

    return {
        "eng_train":     eng_train,
        "eng_test":      eng_test,
        "tgt_train":     tgt_train,
        "tgt_test":      tgt_test,
        "eng_tokenizer": eng_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_translation_artifacts(language: str) -> dict:
    """
    Load preprocessed padded arrays and tokenisers from disk.

    Args:
        language: Target language key.

    Returns:
        Same dict structure as prepare_translation_dataset().
    """
    train_path = get_translation_artifact_path(language, "train")
    test_path  = get_translation_artifact_path(language, "test")

    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                f"  → Run preprocessing notebook first."
            )

    with open(train_path, "rb") as fh:
        train = pickle.load(fh)
    with open(test_path, "rb") as fh:
        test  = pickle.load(fh)

    eng_tokenizer = load_tokenizer(get_eng_tokenizer_path(language))
    tgt_tokenizer = load_tokenizer(get_target_tokenizer_path(language))

    return {
        "eng_train":     train["eng"],
        "eng_test":      test["eng"],
        "tgt_train":     train["tgt"],
        "tgt_test":      test["tgt"],
        "eng_tokenizer": eng_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
    }
