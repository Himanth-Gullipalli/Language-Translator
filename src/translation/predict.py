"""
predict.py  (translation)
──────────────────────────
Greedy-decoding inference for the Seq2Seq NMT model.

Provides:
  • load_model_for_language()  — load & cache encoder/decoder + tokenisers
  • translate()               — translate a single English sentence
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import (
    TRANS_ENG_VOCAB_SIZE,
    TRANS_TARGET_VOCAB_SIZE,
    TRANS_EMBEDDING_DIM,
    TRANS_UNITS,
    TRANS_MAX_ENG_LEN,
    TRANS_MAX_TARGET_LEN,
    get_encoder_weights_path,
    get_decoder_weights_path,
    get_eng_tokenizer_path,
    get_target_tokenizer_path,
    START_TOKEN,
    END_TOKEN,
)
from src.translation.encoder import TranslationEncoder
from src.translation.decoder import TranslationDecoder
from src.translation.preprocessing import load_tokenizer, preprocess_for_inference

# In-memory model cache — avoids reloading on every call
_MODEL_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_language(language: str) -> dict:
    """
    Load (and cache) the encoder, decoder, and tokenisers for a language pair.

    On the first call the model is loaded from disk; subsequent calls return
    the cached instance immediately (no I/O).

    Args:
        language: Target language key (e.g., 'french').

    Returns:
        dict with keys:
            encoder       — TranslationEncoder instance (weights loaded)
            decoder       — TranslationDecoder instance (weights loaded)
            eng_tokenizer — source-language Keras Tokenizer
            tgt_tokenizer — target-language Keras Tokenizer
            idx2word      — reverse mapping {int → str} for target vocabulary

    Raises:
        FileNotFoundError: If model weights or tokenisers are not found on disk.
    """
    if language in _MODEL_CACHE:
        return _MODEL_CACHE[language]

    # ── Tokenisers ───────────────────────────────────────────────────────────
    eng_tokenizer = load_tokenizer(get_eng_tokenizer_path(language))
    tgt_tokenizer = load_tokenizer(get_target_tokenizer_path(language))

    eng_vocab = min(TRANS_ENG_VOCAB_SIZE,    len(eng_tokenizer.word_index) + 1)
    tgt_vocab = min(TRANS_TARGET_VOCAB_SIZE,  len(tgt_tokenizer.word_index) + 1)

    # ── Build model with same hyper-params used during training ──────────────
    encoder = TranslationEncoder(eng_vocab, TRANS_EMBEDDING_DIM, TRANS_UNITS)
    decoder = TranslationDecoder(tgt_vocab, TRANS_EMBEDDING_DIM, TRANS_UNITS, TRANS_UNITS)

    # Build sub-models by running a dummy forward pass (initialises weights)
    dummy_enc_in = tf.zeros((1, TRANS_MAX_ENG_LEN), dtype=tf.int32)
    enc_out_dummy, h_dummy, _ = encoder(dummy_enc_in, training=False)
    dummy_dec_in = tf.zeros((1, 1), dtype=tf.int32)
    decoder(dummy_dec_in, h_dummy, enc_out_dummy, training=False)

    # ── Load weights ─────────────────────────────────────────────────────────
    enc_path = get_encoder_weights_path(language)
    dec_path = get_decoder_weights_path(language)

    for path in [enc_path, dec_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model weights not found: {path}\n"
                f"  → Train the model first using the training notebook."
            )

    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)

    idx2word = {v: k for k, v in tgt_tokenizer.word_index.items()}

    _MODEL_CACHE[language] = {
        "encoder":       encoder,
        "decoder":       decoder,
        "eng_tokenizer": eng_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "idx2word":      idx2word,
    }
    print(f"[INFO] Translation model loaded for language: {language}")
    return _MODEL_CACHE[language]


# ─────────────────────────────────────────────────────────────────────────────
# Greedy Decoding
# ─────────────────────────────────────────────────────────────────────────────

def translate(
    text:     str,
    language: str,
    max_len:  int = TRANS_MAX_TARGET_LEN,
) -> dict:
    """
    Translate a single English sentence to the target language.

    Uses greedy decoding: at each step the token with the highest logit
    is selected as the next input.

    Args:
        text:     Raw English input sentence.
        language: Target language name (e.g., 'french').
        max_len:  Maximum number of tokens to decode.

    Returns:
        dict with keys:
            translation      — Translated string (decoded tokens joined by spaces).
            attention_weights — List of attention weight arrays (one per decode step),
                               each of shape (1, src_len, 1).
    """
    model_data    = load_model_for_language(language)
    encoder       = model_data["encoder"]
    decoder       = model_data["decoder"]
    eng_tokenizer = model_data["eng_tokenizer"]
    tgt_tokenizer = model_data["tgt_tokenizer"]
    idx2word      = model_data["idx2word"]

    # ── Preprocess input ─────────────────────────────────────────────────────
    enc_input = preprocess_for_inference(text, eng_tokenizer, TRANS_MAX_ENG_LEN)
    enc_input = tf.convert_to_tensor(enc_input, dtype=tf.int32)

    # ── Encode ───────────────────────────────────────────────────────────────
    enc_output, state_h, _ = encoder(enc_input, training=False)
    dec_hidden = state_h

    # ── Special token IDs ────────────────────────────────────────────────────
    start_id = tgt_tokenizer.word_index.get(START_TOKEN, 1)
    end_id   = tgt_tokenizer.word_index.get(END_TOKEN,   2)

    dec_input = tf.expand_dims([start_id], axis=0)  # (1, 1)

    # ── Greedy decode loop ───────────────────────────────────────────────────
    result         = []
    attention_list = []

    for _ in range(max_len):
        logits, dec_hidden, attn_weights = decoder(
            dec_input, dec_hidden, enc_output, training=False
        )
        # logits: (1, 1, vocab_size)
        predicted_id = int(tf.argmax(logits[0, 0]).numpy())
        attention_list.append(attn_weights.numpy())   # (1, src_len, 1)

        if predicted_id == end_id:
            break

        word = idx2word.get(predicted_id, "")
        if word and word not in {START_TOKEN, END_TOKEN, "<oov>", ""}:
            result.append(word)

        dec_input = tf.expand_dims([predicted_id], axis=0)

    return {
        "translation":      " ".join(result),
        "attention_weights": attention_list,
    }
