"""Singleton embedding encoder cache.

Provides an LRU-cached SentenceTransformer instance so the
ko-sroberta-multitask model is loaded only once per process regardless
of how many modules call encode_texts().
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL: str = "jhgan/ko-sroberta-multitask"


@contextmanager
def _suppress_noisy_output():
    """Suppress stdout/stderr at the file-descriptor level.

    sentence-transformers and huggingface_hub print LOAD REPORT and
    HF_TOKEN warnings via a mix of Python print(), logging, and
    C-level writes.  Only fd-level redirection silences all of them.
    """
    old_out = os.dup(1)
    old_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_out)
        os.close(old_err)


@lru_cache(maxsize=4)
def get_encoder(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer encoder.

    The first call for a given ``model_name`` loads the model from disk
    (or downloads it). Subsequent calls return the same object without
    re-loading.  Noisy LOAD REPORT and HF_TOKEN warnings are suppressed.

    Args:
        model_name: HuggingFace model identifier or local path.
            Defaults to ``jhgan/ko-sroberta-multitask``.

    Returns:
        Loaded SentenceTransformer instance.

    Examples:
        >>> enc = get_encoder()
        >>> enc is get_encoder()  # same instance
        True
    """
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)

    with _suppress_noisy_output():
        encoder = SentenceTransformer(model_name)
    return encoder


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Encode a list of texts into embedding vectors.

    Uses the cached encoder to avoid reloading the model.

    Args:
        texts: Non-empty list of strings to encode.
        model_name: Encoder model to use (cached by get_encoder).

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dim).

    Raises:
        ValueError: If ``texts`` is empty.

    Examples:
        >>> vecs = encode_texts(["세포막은 인지질 이중층이다"])
        >>> vecs.shape[0]
        1
    """
    if not texts:
        raise ValueError("texts is empty in encode_texts(). Provide at least one string to encode.")
    encoder = get_encoder(model_name)
    return encoder.encode(texts, convert_to_numpy=True)
