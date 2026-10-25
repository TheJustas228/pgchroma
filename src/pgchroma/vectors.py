"""Embedding transport: JSON float lists or base64 little-endian float32 (Chroma's fast path)."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np

from .errors import InvalidArgumentError


def _one(item: Any) -> np.ndarray:
    if isinstance(item, str):
        try:
            raw = base64.b64decode(item, validate=True)
        except (binascii.Error, ValueError) as e:
            raise InvalidArgumentError(f"Invalid base64 embedding: {e}") from e
        if len(raw) % 4:
            raise InvalidArgumentError("Invalid base64 embedding: length is not a multiple of 4")
        return np.frombuffer(raw, dtype="<f4")
    if isinstance(item, list):
        try:
            return np.asarray(item, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise InvalidArgumentError(f"Invalid embedding: {e}") from e
    raise InvalidArgumentError(
        f"Expected an embedding to be a list of numbers, got {type(item).__name__}"
    )


def decode_embeddings(items: list | None) -> np.ndarray | None:
    """Return an (n, dim) float32 array, or None when no embeddings were sent."""
    if items is None:
        return None
    if not isinstance(items, list):
        raise InvalidArgumentError("Expected embeddings to be a list")
    if not items:
        return np.zeros((0, 0), dtype=np.float32)
    rows = [_one(i) for i in items]
    dims = {r.shape[0] if r.ndim == 1 else -1 for r in rows}
    if len(dims) != 1 or -1 in dims:
        dims_list = [r.shape[0] for r in rows]
        raise InvalidArgumentError(
            f"Collection expecting embedding with dimension of {dims_list[0]}, got {next(d for d in dims_list if d != dims_list[0])}"
        )
    if 0 in dims:
        raise InvalidArgumentError("Expected embeddings to be non-empty")
    out = np.stack(rows).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise InvalidArgumentError("Embeddings must not contain NaN or infinite values")
    return out
