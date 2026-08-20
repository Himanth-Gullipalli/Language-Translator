"""
config.py
─────────
Central configuration for the Neural Machine Translation & Sentiment Analysis project.
All file paths, hyperparameters, and shared constants live here.
Import this module at the top of every other module instead of hard-coding paths.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Base Directory  (resolves to: Neural_Machine_Translation/)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Top-level Directories
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(BASE_DIR, "datasets")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# Translation sub-directories
TRANSLATION_DATASET_DIR  = os.path.join(DATASET_DIR,  "translation")
TRANSLATION_MODEL_DIR    = os.path.join(MODEL_DIR,    "translation")
TRANSLATION_ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, "translation")

# Sentiment sub-directories
SENTIMENT_DATASET_DIR    = os.path.join(DATASET_DIR,  "sentiment")
SENTIMENT_MODEL_DIR      = os.path.join(MODEL_DIR,    "sentiment")
SENTIMENT_ARTIFACT_DIR   = os.path.join(ARTIFACT_DIR, "sentiment")

# ─────────────────────────────────────────────────────────────────────────────
# Supported Languages
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["french", "spanish", "german", "hindi"]

# Map: language → CSV filename under datasets/translation/
LANGUAGE_CSV_MAP = {
    "french":  "english_french.csv",
    "spanish": "english_spanish.csv",
    "german":  "english_german.csv",
    "hindi":   "english_hindi.csv",
}

# Map: language → target column name inside the CSV
LANGUAGE_COL_MAP = {
    "french":  "french",
    "spanish": "spanish",
    "german":  "german",
    "hindi":   "hindi",
}

# ─────────────────────────────────────────────────────────────────────────────
# Translation Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TRANS_ENG_VOCAB_SIZE    = 15_000   # English source vocabulary cap
TRANS_TARGET_VOCAB_SIZE = 15_000   # Target language vocabulary cap
TRANS_EMBEDDING_DIM     = 256      # Word embedding dimension
TRANS_UNITS             = 512      # Encoder LSTM / Decoder GRU units
TRANS_BATCH_SIZE        = 64       # Training batch size
TRANS_EPOCHS            = 20       # Maximum training epochs
TRANS_MAX_ENG_LEN       = 30       # Max tokens per English sentence
TRANS_MAX_TARGET_LEN    = 30       # Max tokens per target sentence (incl. special tokens)

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
SENT_VOCAB_SIZE    = 20_000   # Vocabulary cap for sentiment tokenizer
SENT_EMBEDDING_DIM = 128      # Word embedding dimension
SENT_LSTM_UNITS    = 128      # BiLSTM units per direction
SENT_DROPOUT_RATE  = 0.3      # Dropout probability
SENT_BATCH_SIZE    = 64       # Training batch size
SENT_EPOCHS        = 15       # Maximum training epochs
SENT_MAX_LEN       = 100      # Max tokens per input review/sentence

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Label Map  (integer index → human-readable label)
# Order must match LabelEncoder's alphabetical encoding:
#   0 → negative, 1 → neutral, 2 → positive
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

# ─────────────────────────────────────────────────────────────────────────────
# Emotion Keyword Dictionary  (rule-based emotion overlay)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_KEYWORDS = {
    "joy": [
        "happy", "joy", "joyful", "excited", "wonderful", "fantastic",
        "love", "great", "amazing", "delighted", "glad", "cheerful",
        "thrilled", "blissful", "elated", "pleased",
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "miserable", "crying", "tears",
        "heartbroken", "grief", "sorrow", "lonely", "melancholy",
        "devastated", "disappointed", "hopeless",
    ],
    "anger": [
        "angry", "furious", "rage", "hate", "mad", "irritated",
        "frustrated", "outraged", "disgusted", "annoyed", "bitter",
        "resentful", "hostile",
    ],
    "fear": [
        "afraid", "scared", "fearful", "terrified", "anxious",
        "nervous", "worried", "panic", "dread", "horrified",
        "uneasy", "apprehensive",
    ],
    "surprise": [
        "surprised", "shocked", "astonished", "amazed", "unexpected",
        "sudden", "wow", "unbelievable", "incredible", "stunning",
        "startled",
    ],
    "disgust": [
        "disgusting", "gross", "horrible", "awful", "terrible",
        "nasty", "revolting", "repulsive", "vile", "loathsome",
    ],
    "trust": [
        "trust", "reliable", "honest", "confident", "sure",
        "believe", "faith", "sincere", "dependable", "loyal",
    ],
    "anticipation": [
        "hope", "expect", "anticipate", "looking forward", "eager",
        "await", "curious", "excited about", "can't wait",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Special Tokens
# ─────────────────────────────────────────────────────────────────────────────
START_TOKEN = "<start>"
END_TOKEN   = "<end>"

# ─────────────────────────────────────────────────────────────────────────────
# Model Path Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_encoder_weights_path(lang: str) -> str:
    """Path: models/translation/encoder_{lang}.weights.h5"""
    return os.path.join(TRANSLATION_MODEL_DIR, f"encoder_{lang}.weights.h5")


def get_decoder_weights_path(lang: str) -> str:
    """Path: models/translation/decoder_{lang}.weights.h5"""
    return os.path.join(TRANSLATION_MODEL_DIR, f"decoder_{lang}.weights.h5")


def get_eng_tokenizer_path(lang: str) -> str:
    """Path: models/translation/english_{lang}_tokenizer.pkl"""
    return os.path.join(TRANSLATION_MODEL_DIR, f"english_{lang}_tokenizer.pkl")


def get_target_tokenizer_path(lang: str) -> str:
    """Path: models/translation/{lang}_tokenizer.pkl"""
    return os.path.join(TRANSLATION_MODEL_DIR, f"{lang}_tokenizer.pkl")


def get_translation_artifact_path(lang: str, split: str) -> str:
    """
    Path: artifacts/translation/{lang}_{split}.pkl
    split ∈ {'train', 'test'}
    """
    return os.path.join(TRANSLATION_ARTIFACT_DIR, f"{lang}_{split}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Paths
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_DATASET_PATH   = os.path.join(SENTIMENT_DATASET_DIR,  "sentiment.csv")
SENTIMENT_MODEL_PATH     = os.path.join(SENTIMENT_MODEL_DIR,    "sentiment_model.keras")
SENTIMENT_TOKENIZER_PATH = os.path.join(SENTIMENT_MODEL_DIR,    "tokenizer.pkl")
SENTIMENT_LABEL_ENC_PATH = os.path.join(SENTIMENT_MODEL_DIR,    "label_encoder.pkl")
SENTIMENT_TRAIN_ARTIFACT = os.path.join(SENTIMENT_ARTIFACT_DIR, "processed_train.pkl")
SENTIMENT_TEST_ARTIFACT  = os.path.join(SENTIMENT_ARTIFACT_DIR, "processed_test.pkl")
