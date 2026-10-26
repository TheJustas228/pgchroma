"""Chroma's Search API (`POST .../search`) on Postgres.

Supported: filters (`where` syntax plus `#id` / `#document`), ranking expressions built from
`Knn` (distance or 0-based rank, `limit`, `default`) over the dense embedding or over sparse
vectors stored in metadata (with Chroma's BM25 weighting when the schema asks for it), combined
with `$val`, `$sum`, `$sub`, `$mul`, `$div`, `$abs`, `$exp`, `$log`, `$max`, `$min` (which is how
the client encodes `Rrf`), `limit`/`offset`, and `select`.

Not yet: `group_by`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from psycopg import sql

from .errors import InvalidArgumentError, NotSupportedError
from .filters import compile_search_filter
from .store import CollectionRef, Store, _as_array

EMBEDDING_KEY = "#embedding"
_SPECIAL_SELECT = {"#document", "#embedding", "#metadata", "#score", "#id"}


class _Knn:
    def __init__(self, spec: dict):
        if not isinstance(spec, dict) or "query" not in spec:
            raise InvalidArgumentError(f"Invalid $knn expression: {spec!r}")
        self.key = spec.get("key", EMBEDDING_KEY)
        query = spec["query"]
        self.sparse: tuple[list[int], list[float]] | None = None
        if self.key == EMBEDDING_KEY:
            if not isinstance(query, list):
                raise InvalidArgumentError("Knn over #embedding expects a dense query vector")
            self.query = np.asarray([query], dtype=np.float32)
        elif not isinstance(self.key, str) or self.key.startswith("#"):
            raise InvalidArgumentError(
                f"Knn key must be #embedding or a metadata field, got {self.key!r}"
            )
        else:
            self.sparse = _sparse_query(query)
        self.limit = int(spec.get("limit", 16))
        if self.limit < 1:
            raise InvalidArgumentError("Knn limit must be at least 1")
        self.default = spec.get("default")
        self.return_rank = bool(spec.get("return_rank", False))
        self.scores: dict[str, float] = {}


def _sparse_query(query: Any) -> tuple[list[int], list[float]]:
    if not isinstance(query, dict) or query.get("#type") != "sparse_vector":
        raise InvalidArgumentError("Knn over a metadata field expects a sparse vector query")
    indices, values = query.get("indices"), query.get("values")
    if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
        raise InvalidArgumentError("A sparse vector needs indices and values of the same length")
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
        raise InvalidArgumentError("Sparse vector indices must be non-negative integers")
    try:
        floats = [float(v) for v in values]
    except (TypeError, ValueError):
        raise InvalidArgumentError("Sparse vector values must be numbers") from None
    if not all(math.isfinite(v) for v in floats):
        raise InvalidArgumentError("Sparse vector values must be finite")
    return indices, floats


def _uses_bm25(schema: dict | None, key: str) -> bool:
    """Chroma's rule: the key's own sparse index config decides; otherwise the schema default."""

    def flag(value_types: Any) -> Any:
        index = ((value_types or {}).get("sparse_vector") or {}).get("sparse_vector_index") or {}
        return (index.get("config") or {}).get("bm25")

    own = flag(((schema or {}).get("keys") or {}).get(key))
    return bool(own if own is not None else flag((schema or {}).get("defaults")))


def _compile_rank(node: Any, knns: list[_Knn]) -> Callable[[str], float | None]:
    """Turn a rank expression into a function of the record id (None = record excluded)."""
    if not isinstance(node, dict) or len(node) != 1:
        raise InvalidArgumentError(f"Invalid rank expression: {node!r}")
    op, arg = next(iter(node.items()))
    if op == "$knn":
        knn = _Knn(arg)
        knns.append(knn)
        return lambda rid: knn.scores.get(rid, knn.default)
    if op == "$val":
        value = float(arg)
        return lambda rid: value
    if op in ("$sum", "$mul", "$max", "$min"):
        if not isinstance(arg, list) or not arg:
            raise InvalidArgumentError(f"{op} expects a non-empty list")
        parts = [_compile_rank(a, knns) for a in arg]
        combine = {"$sum": sum, "$mul": math.prod, "$max": max, "$min": min}[op]

        def nary(rid):
            values = [p(rid) for p in parts]
            return None if any(v is None for v in values) else combine(values)

        return nary
    if op in ("$sub", "$div"):
        if not isinstance(arg, dict) or set(arg) != {"left", "right"}:
            raise InvalidArgumentError(f"{op} expects left and right operands")
        left, right = _compile_rank(arg["left"], knns), _compile_rank(arg["right"], knns)

        def binary(rid):
            a, b = left(rid), right(rid)
            if a is None or b is None:
                return None
            if op == "$sub":
                return a - b
            return a / b if b != 0 else math.inf

        return binary
    if op in ("$abs", "$exp", "$log"):
        inner = _compile_rank(arg, knns)
        fn = {"$abs": abs, "$exp": math.exp, "$log": lambda v: math.log(v) if v > 0 else -math.inf}[
            op
        ]
        return lambda rid: None if (v := inner(rid)) is None else fn(v)
    raise NotSupportedError(f"Rank operator {op} is not supported by pgchroma")


def _parse_limit(spec: Any) -> tuple[int, int | None]:
    if spec is None:
        return 0, None
    if not isinstance(spec, dict):
        raise InvalidArgumentError(f"Invalid limit: {spec!r}")
    offset = int(spec.get("offset") or 0)
    limit = spec.get("limit")
    return offset, None if limit is None else int(limit)


async def _fetch(store: Store, coll: CollectionRef, ids: list[str]) -> dict[str, tuple]:
    if not ids:
        return {}
    meta = "CASE WHEN metadata IS NULL AND uri IS NOT NULL THEN '{{}}'::jsonb ELSE metadata END"
    async with store._tx() as conn:
        rows = await (
            await conn.execute(
                sql.SQL(
                    "SELECT id, seq, document, embedding, " + meta + " FROM {} WHERE id = ANY(%s)"
                ).format(coll.table),
                (ids,),
            )
        ).fetchall()
    return {r[0]: r for r in rows}


async def _one(store: Store, coll: CollectionRef, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise InvalidArgumentError("Each search must be an object")
    if payload.get("group_by"):
        raise NotSupportedError("group_by is not supported by pgchroma yet")
    cond_sql, params = compile_search_filter(payload.get("filter"))
    cond = sql.SQL(cond_sql)
    offset, limit = _parse_limit(payload.get("limit"))
    keys = list((payload.get("select") or {}).get("keys") or [])

    knns: list[_Knn] = []
    rank = payload.get("rank")
    score_of = _compile_rank(rank, knns) if rank is not None else None

    if score_of is None:
        # Filter only: records in insertion order, no scores.
        async with store._tx() as conn:
            rows = await (
                await conn.execute(
                    sql.SQL("SELECT id FROM {} WHERE {} ORDER BY seq LIMIT %s OFFSET %s").format(
                        coll.table, cond
                    ),
                    [*params, limit, offset],
                )
            ).fetchall()
        ordered = [(r[0], None) for r in rows]
    else:
        candidates: dict[str, None] = {}
        for knn in knns:
            if knn.sparse is not None:
                rows = await store.sparse_knn(
                    coll,
                    knn.key,
                    *knn.sparse,
                    knn.limit,
                    cond,
                    params,
                    bm25=_uses_bm25(coll.schema, knn.key),
                )
            elif coll.dimension is None:
                continue  # no records yet
            elif knn.query.shape[1] != coll.dimension:
                raise InvalidArgumentError(
                    f"Collection expecting embedding with dimension of {coll.dimension}, got {knn.query.shape[1]}"
                )
            else:
                rows = await store.knn(
                    coll, knn.query, knn.limit, cond, params, payload.get("filter") is not None, []
                )
            for position, row in enumerate(rows):
                knn.scores[row[1]] = float(position) if knn.return_rank else float(row[2])
                candidates.setdefault(row[1])
        scored = [(rid, score_of(rid)) for rid in candidates]
        scored = [(rid, s) for rid, s in scored if s is not None and not math.isnan(s)]
        seqs = await _fetch(store, coll, [rid for rid, _ in scored])
        scored.sort(key=lambda t: (t[1], seqs[t[0]][1]))
        ordered = scored[offset : None if limit is None else offset + limit]

    ids = [rid for rid, _ in ordered]
    want_meta_fields = [k for k in keys if k not in _SPECIAL_SELECT]
    needs_rows = (
        any(k in keys for k in ("#document", "#embedding", "#metadata")) or want_meta_fields
    )
    rows_by_id = await _fetch(store, coll, ids) if needs_rows else {}

    def column(fn):
        return [fn(rows_by_id[rid]) for rid in ids]

    out: dict[str, Any] = {"ids": ids}
    out["documents"] = column(lambda r: r[2]) if "#document" in keys else None
    out["embeddings"] = (
        [np.asarray(_as_array(r), dtype=np.float32) for r in column(lambda r: r[3])]
        if "#embedding" in keys
        else None
    )
    if "#metadata" in keys:
        out["metadatas"] = column(lambda r: r[4])
    elif want_meta_fields:
        out["metadatas"] = column(
            lambda r: {k: r[4][k] for k in want_meta_fields if r[4] and k in r[4]} or None
        )
    else:
        out["metadatas"] = None
    out["scores"] = [s for _, s in ordered] if "#score" in keys else None
    out["select"] = keys
    return out


async def search(store: Store, coll: CollectionRef, body: dict) -> dict:
    searches = body.get("searches")
    if not isinstance(searches, list) or not searches:
        raise InvalidArgumentError("Expected `searches` to be a non-empty list")
    results = [await _one(store, coll, s) for s in searches]
    return {
        field: [r[field] for r in results]
        for field in ("ids", "documents", "embeddings", "metadatas", "scores", "select")
    }
