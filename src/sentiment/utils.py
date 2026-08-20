"""
utils.py  (sentiment)
──────────────────────
Utility helpers for the sentiment analysis module:
  • Training curve plots
  • Confusion matrix visualisation
  • Confidence bar formatter
  • Sentiment and emotion colour/emoji helpers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Training History
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history) -> plt.Figure:
    """
    Plot accuracy and loss curves for both training and validation sets.

    Args:
        history: Keras History object returned by model.fit().

    Returns:
        Matplotlib Figure with two side-by-side subplots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2, color="#3498DB")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2, color="#E74C3C", linestyle="--")
    axes[0].set_title("Model Accuracy",  fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch",   fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Loss subplot
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2, color="#3498DB")
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, color="#E74C3C", linestyle="--")
    axes[1].set_title("Model Loss",  fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch",  fontsize=11)
    axes[1].set_ylabel("Loss",   fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Sentiment Model — Training History", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:      list,
    y_pred:      list,
    class_names: list,
    title:       str  = "Confusion Matrix",
    normalise:   bool = False,
) -> plt.Figure:
    """
    Plot a styled confusion matrix as a seaborn heatmap.

    Args:
        y_true:      True integer labels.
        y_pred:      Predicted integer labels.
        class_names: List of class name strings for axis labels.
        title:       Plot title.
        normalise:   If True, show percentages instead of raw counts.

    Returns:
        Matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalise:
        cm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title,            fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted",     fontsize=12)
    ax.set_ylabel("True",          fontsize=12)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_confidence_bar(confidence: float, width: int = 40) -> str:
    """Return an ASCII progress bar representing model confidence."""
    filled = int(confidence * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {confidence * 100:.1f}%"


def get_sentiment_color(label: str) -> str:
    """Return a hex colour for a given sentiment label string."""
    colours = {
        "Positive": "#2ECC71",
        "Negative": "#E74C3C",
        "Neutral":  "#F39C12",
    }
    return colours.get(label, "#95A5A6")


def get_emotion_emoji(emotion: str) -> str:
    """Return an emoji for a given emotion name."""
    emojis = {
        "joy":          "😊",
        "sadness":      "😢",
        "anger":        "😠",
        "fear":         "😨",
        "surprise":     "😲",
        "disgust":      "🤢",
        "trust":        "🤝",
        "anticipation": "🌟",
        "neutral":      "😐",
    }
    return emojis.get(emotion, "❓")


def get_sentiment_icon(label: str) -> str:
    """Return an emoji icon for a given sentiment label."""
    icons = {
        "Positive": "✅",
        "Negative": "❌",
        "Neutral":  "⚖️",
    }
    return icons.get(label, "❓")
