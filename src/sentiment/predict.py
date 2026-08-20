"""
predict.py  (sentiment)
────────────────────────
Inference for the sentiment analysis model.

Provides:
  • load_sentiment_model()  — load & cache model + tokeniser
  • predict_sentiment()     — neural sentiment classification
  • predict_emotion()       — rule-based emotion detection
  • full_analysis()         — combined result for a single text
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    SENTIMENT_MODEL_PATH,
    SENTIMENT_TOKENIZER_PATH,
    SENTIMENT_LABELS,
    EMOTION_KEYWORDS,
    SENT_MAX_LEN,
)
from src.sentiment.preprocessing import clean_text, load_tokenizer, texts_to_padded

# In-memory cache
_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sentiment_model() -> dict:
    """
    Load (and cache) the sentiment Keras model and tokeniser from disk.

    Returns:
        dict with keys: model, tokenizer

    Raises:
        FileNotFoundError: If the model or tokeniser files are missing.
    """
    if "model" in _CACHE:
        return _CACHE

    for path, label in [
        (SENTIMENT_MODEL_PATH,     "Sentiment model"),
        (SENTIMENT_TOKENIZER_PATH, "Sentiment tokenizer"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "  → Train the model first using notebooks/sentiment/."
            )

    _CACHE["model"]     = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
    _CACHE["tokenizer"] = load_tokenizer(SENTIMENT_TOKENIZER_PATH)
    print("[INFO] Sentiment model loaded successfully.")
    return _CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_sentiment(text: str) -> dict:
    """
    Classify the sentiment of the input text.

    Args:
        text: Raw English input string.

    Returns:
        dict with keys:
            label         — Sentiment label ('Positive', 'Negative', 'Neutral').
            label_index   — Integer class index (0/1/2).
            confidence    — Model confidence for the predicted class [0, 1].
            probabilities — {label_str: probability_pct} for all classes.
    """
    cache     = load_sentiment_model()
    model     = cache["model"]
    tokenizer = cache["tokenizer"]

    cleaned = clean_text(text)
    padded  = texts_to_padded(tokenizer, [cleaned], maxlen=SENT_MAX_LEN)
    probs   = model.predict(padded, verbose=0)[0]   # (num_classes,)

    label_index = int(np.argmax(probs))
    label       = SENTIMENT_LABELS.get(label_index, "Unknown")
    confidence  = float(probs[label_index])

    probabilities = {
        SENTIMENT_LABELS.get(i, str(i)): round(float(p) * 100, 2)
        for i, p in enumerate(probs)
    }

    return {
        "label":         label,
        "label_index":   label_index,
        "confidence":    confidence,
        "probabilities": probabilities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Emotion Detection (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def predict_emotion(text: str) -> dict:
    """
    Detect the dominant emotion using keyword-matching over EMOTION_KEYWORDS.

    This rule-based layer complements the neural classifier by providing
    fine-grained emotional insight (joy, sadness, anger, fear, surprise, etc.).

    Args:
        text: Raw input string.

    Returns:
        dict with keys:
            dominant_emotion — Top emotion string (or 'neutral' if none found).
            emotion_scores   — {emotion: keyword_match_count} for all emotions.
            description      — Human-readable description with emoji.
    """
    text_lower = text.lower()

    scores: dict = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in text_lower)

    dominant = max(scores, key=scores.get)
    if scores[dominant] == 0:
        dominant = "neutral"

    descriptions = {
        "joy":          "😊 The text conveys joy and happiness.",
        "sadness":      "😢 The text expresses sadness or grief.",
        "anger":        "😠 The text indicates anger or frustration.",
        "fear":         "😨 The text suggests fear or anxiety.",
        "surprise":     "😲 The text reflects surprise or astonishment.",
        "disgust":      "🤢 The text communicates disgust or revulsion.",
        "trust":        "🤝 The text conveys trust and reliability.",
        "anticipation": "🌟 The text shows anticipation or eagerness.",
        "neutral":      "😐 No strong emotion detected — the text appears neutral.",
    }

    return {
        "dominant_emotion": dominant,
        "emotion_scores":   scores,
        "description":      descriptions.get(dominant, ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined Analysis
# ─────────────────────────────────────────────────────────────────────────────

def full_analysis(text: str) -> dict:
    """
    Run both sentiment classification and emotion detection on a single text.

    Args:
        text: Raw English input string.

    Returns:
        Merged dict containing all keys from predict_sentiment() and
        predict_emotion() (emotion keys do not conflict with sentiment keys).
    """
    sentiment = predict_sentiment(text)
    emotion   = predict_emotion(text)
    return {**sentiment, **emotion}
