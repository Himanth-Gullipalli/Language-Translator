"""
utils.py  (translation)
────────────────────────
General helper functions for the translation module:
  • Index ↔ word mappings
  • Sequence-to-sentence decoding
  • Attention heatmap plotting
  • Metrics table formatting
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary Helpers
# ─────────────────────────────────────────────────────────────────────────────

def index_to_word(tokenizer: Tokenizer) -> dict:
    """
    Build a reverse mapping from token index to word string.

    Args:
        tokenizer: Fitted Keras Tokenizer.

    Returns:
        dict  {int: str}
    """
    return {v: k for k, v in tokenizer.word_index.items()}


def sequence_to_sentence(
    sequence:      list,
    tokenizer:     Tokenizer,
    skip_special:  bool = True,
) -> str:
    """
    Convert a list of integer token IDs back to a readable sentence.

    Args:
        sequence:     List of integer token IDs.
        tokenizer:    Fitted Keras Tokenizer for the target language.
        skip_special: If True, filter out <start>, <end>, <oov>, and padding (0).

    Returns:
        Decoded sentence string with tokens joined by spaces.
    """
    idx2word = index_to_word(tokenizer)
    special  = {"<start>", "<end>", "<oov>"}
    words    = []

    for idx in sequence:
        if idx == 0:             # padding
            continue
        word = idx2word.get(int(idx), "")
        if skip_special and word in special:
            continue
        if word:
            words.append(word)

    return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# Attention Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention(
    attention:     np.ndarray,
    source_tokens: list,
    target_tokens: list,
    title:         str = "Bahdanau Attention Heatmap",
    cmap:          str = "viridis",
) -> plt.Figure:
    """
    Plot a Bahdanau attention heatmap.

    Args:
        attention:     2-D array of shape (tgt_len, src_len).
        source_tokens: List of source-language tokens (x-axis).
        target_tokens: List of predicted target-language tokens (y-axis).
        title:         Plot title.
        cmap:          Matplotlib colormap name.

    Returns:
        Matplotlib Figure object (caller is responsible for plt.show() or saving).
    """
    fig, ax = plt.subplots(
        figsize=(max(8, len(source_tokens) + 2),
                 max(6, len(target_tokens) + 2))
    )

    img = ax.matshow(attention, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=90, fontsize=9)
    ax.set_yticklabels(target_tokens, fontsize=9)

    ax.set_xlabel("Source (English)", fontsize=11, labelpad=10)
    ax.set_ylabel("Target",           fontsize=11, labelpad=10)
    ax.set_title(title,               fontsize=13, pad=20)

    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_table(metrics: dict) -> str:
    """
    Format a metrics dictionary as a plain-text aligned table.

    Args:
        metrics: dict produced by compute_all_metrics(), values as percentages.

    Returns:
        Multi-line string table.
    """
    name_map = {
        "bleu":           "BLEU (%)",
        "rouge1":         "ROUGE-1 (%)",
        "rouge2":         "ROUGE-2 (%)",
        "rougeL":         "ROUGE-L (%)",
        "token_accuracy": "Token Accuracy (%)",
    }
    col_width = 22
    lines = [
        f"{'Metric':<{col_width}} {'Score':>8}",
        "─" * (col_width + 10),
    ]
    for key, val in metrics.items():
        label = name_map.get(key, key)
        lines.append(f"{label:<{col_width}} {val:>8.2f}")
    return "\n".join(lines)


def get_attention_matrix(attention_weights: list) -> np.ndarray:
    """
    Stack per-step attention weight arrays into a 2-D (tgt_len, src_len) matrix.

    Args:
        attention_weights: List of arrays, each shape (1, src_len, 1).

    Returns:
        NumPy array of shape (tgt_len, src_len).
    """
    # Each element: (1, src_len, 1) → (src_len,)
    steps = [w[0, :, 0] for w in attention_weights]
    return np.array(steps)   # (tgt_len, src_len)
