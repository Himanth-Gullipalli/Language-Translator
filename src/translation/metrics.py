"""
metrics.py  (translation)
──────────────────────────
Translation quality evaluation metrics:
  • BLEU   — bilingual evaluation understudy (sentence & corpus level)
  • ROUGE  — recall-oriented understudy for gisting evaluation
  • Token Accuracy — fraction of correctly predicted tokens (same position)
"""

import numpy as np
from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction,
)
from rouge_score import rouge_scorer as _rouge_scorer


# One shared smoother for all BLEU computations
_SMOOTHER = SmoothingFunction().method4


# ─────────────────────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentence_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute sentence-level BLEU score with method-4 smoothing.

    Args:
        reference:  Ground-truth translation string.
        hypothesis: Model-generated translation string.

    Returns:
        Float BLEU score in [0, 1].
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTHER)


def compute_corpus_bleu(references: list, hypotheses: list) -> float:
    """
    Compute corpus-level BLEU score.

    Args:
        references:  List of reference strings.
        hypotheses:  List of hypothesis strings (same order).

    Returns:
        Float BLEU score in [0, 1].
    """
    ref_tokens = [[r.split()] for r in references]
    hyp_tokens = [h.split()   for h in hypotheses]
    return corpus_bleu(ref_tokens, hyp_tokens, smoothing_function=_SMOOTHER)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        reference:  Ground-truth string.
        hypothesis: Model-generated string.

    Returns:
        dict with keys: rouge1, rouge2, rougeL  (float F1, each in [0, 1]).
    """
    scorer = _rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 6),
        "rouge2": round(scores["rouge2"].fmeasure, 6),
        "rougeL": round(scores["rougeL"].fmeasure, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Token Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_accuracy(reference: str, hypothesis: str) -> float:
    """
    Token-level accuracy: fraction of tokens at the same position that match.

    The denominator is the length of the longer sequence to penalise
    sequences that are too short or too long.

    Args:
        reference:  Ground-truth translation.
        hypothesis: Model-generated translation.

    Returns:
        Float accuracy in [0, 1].
    """
    ref_tok = reference.split()
    hyp_tok = hypothesis.split()
    max_len = max(len(ref_tok), len(hyp_tok), 1)
    matches = sum(r == h for r, h in zip(ref_tok, hyp_tok))
    return matches / max_len


# ─────────────────────────────────────────────────────────────────────────────
# Combined
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute all translation metrics in one call.

    All scores are returned as percentages (multiplied by 100) for readability.

    Args:
        reference:  Ground-truth translation string.
        hypothesis: Model-generated translation string.

    Returns:
        dict with keys: bleu, rouge1, rouge2, rougeL, token_accuracy
        (values are percentages, e.g. 45.23 means 45.23 %).
    """
    bleu     = compute_sentence_bleu(reference, hypothesis)
    rouge    = compute_rouge(reference, hypothesis)
    accuracy = compute_token_accuracy(reference, hypothesis)

    return {
        "bleu":           round(bleu     * 100, 2),
        "rouge1":         round(rouge["rouge1"] * 100, 2),
        "rouge2":         round(rouge["rouge2"] * 100, 2),
        "rougeL":         round(rouge["rougeL"] * 100, 2),
        "token_accuracy": round(accuracy * 100, 2),
    }
