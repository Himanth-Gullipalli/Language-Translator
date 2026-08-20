"""
metrics.py  (sentiment)
────────────────────────
Evaluation metrics for the sentiment classification model:
  • Classification report (precision, recall, F1 per class)
  • Confusion matrix
  • Overall accuracy and macro F1
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_report(
    y_true:      list,
    y_pred:      list,
    class_names: list = None,
) -> str:
    """
    Generate a full per-class classification report.

    Args:
        y_true:       True integer labels.
        y_pred:       Predicted integer labels.
        class_names:  Optional list of class name strings for display.

    Returns:
        Formatted multi-line string report.
    """
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )


def compute_confusion_matrix(y_true: list, y_pred: list) -> np.ndarray:
    """Return confusion matrix as a 2-D NumPy array."""
    return confusion_matrix(y_true, y_pred)


def compute_accuracy(y_true: list, y_pred: list) -> float:
    """Return overall accuracy as a float in [0, 1]."""
    return accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: list, y_pred: list) -> float:
    """Return macro-averaged F1 score as a float in [0, 1]."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def predictions_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convert probability arrays (batch, num_classes) to predicted class indices."""
    return np.argmax(probs, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list = None,
) -> dict:
    """
    Evaluate a trained Keras sentiment model on the test set.

    Args:
        model:       Trained tf.keras model.
        X_test:      Padded test input sequences, shape (N, max_len).
        y_test:      True integer labels, shape (N,).
        class_names: Optional list of label strings (e.g. ['Negative', 'Neutral', 'Positive']).

    Returns:
        dict with keys:
            accuracy         — float overall accuracy
            macro_f1         — float macro F1 score
            report           — string classification report
            confusion_matrix — 2-D np.ndarray
    """
    probs  = model.predict(X_test, verbose=0)
    y_pred = predictions_from_probs(probs)

    acc    = compute_accuracy(y_test, y_pred)
    f1     = compute_macro_f1(y_test, y_pred)
    report = compute_classification_report(y_test, y_pred, class_names)
    cm     = compute_confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:\n")
    print(report)

    return {
        "accuracy":         round(acc, 6),
        "macro_f1":         round(f1, 6),
        "report":           report,
        "confusion_matrix": cm,
    }
