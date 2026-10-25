"""Compile Chroma `where` / `where_document` filters into parameterized SQL.

The generated expressions reference two columns, `metadata` (jsonb) and `document` (text),
and use `%s` placeholders. Semantics follow Chroma 1.5 (see tests/test_filters.py):

* numbers: an int operand also matches float values numerically; a float operand compared
  with *integer* metadata is first truncated toward zero (Chroma's behavior);
* bools never equal numbers, strings never equal numbers;
* `$ne`, `$nin`, `$not_contains` also match records that lack the key;
* `$contains` / `$not_contains` on a metadata key test membership in a list value.
"""

from __future__ import annotations

import math
from typing import Any

import orjson

from .errors import InvalidArgumentError

_COMPARE = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
_WHERE_OPS = ("$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin", "$contains", "$not_contains")
_DOC_OPS = ("$contains", "$not_contains", "$regex", "$not_regex")

# A jsonb number written without '.', 'e' or 'E' is an integer (Chroma's int metadata type).
_IS_FLOAT_TEXT = "(metadata ->> %s) ~ '[.eE]'"


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


def _jsonb(value: Any) -> str:
    return orjson.dumps(value).decode()


def _int_operand(v: int | float) -> int:
    return v if isinstance(v, int) else math.trunc(v)


def _num_case(key: str, float_sql: str, float_params: list, int_sql: str, int_params: list):
    """CASE that only casts after checking the jsonb type, so no row can raise a cast error."""
    sql = (
        f"(CASE WHEN jsonb_typeof(metadata -> %s) = 'number' THEN "
        f"CASE WHEN {_IS_FLOAT_TEXT} THEN {float_sql} ELSE {int_sql} END ELSE FALSE END)"
    )
    return sql, [key, key, *float_params, *int_params]


def _eq(key: str, value: Any) -> tuple[str, list]:
    if isinstance(value, (str, bool)):
        # Containment is exact for strings and bools and can use the GIN index.
        return "(metadata @> %s::jsonb)", [_jsonb({key: value})]
    iop, fop = _int_operand(value), float(value)
    exact, params = _num_case(
        key,
        "(metadata ->> %s)::float8 = %s::float8",
        [key, fop],
        "(metadata ->> %s)::numeric = %s::numeric",
        [key, iop],
    )
    prefilter = "(metadata @> %s::jsonb OR metadata @> %s::jsonb)"
    return f"({prefilter} AND {exact})", [_jsonb({key: iop}), _jsonb({key: fop}), *params]


def _in(key: str, values: list) -> tuple[str, list]:
    first = values[0]
    if isinstance(first, bool):
        return "(metadata -> %s) = ANY(%s::jsonb[])", [key, [_jsonb(v) for v in values]]
    if isinstance(first, str):
        return (
            "(jsonb_typeof(metadata -> %s) = 'string' AND (metadata ->> %s) = ANY(%s::text[]))",
            [key, key, list(values)],
        )
    return _num_case(
        key,
        "(metadata ->> %s)::float8 = ANY(%s::float8[])",
        [key, [float(v) for v in values]],
        "(metadata ->> %s)::numeric = ANY(%s::numeric[])",
        [key, [_int_operand(v) for v in values]],
    )


def _list_contains(key: str, value: Any) -> tuple[str, list]:
    if isinstance(value, (str, bool)):
        return (
            "(jsonb_typeof(metadata -> %s) = 'array' AND (metadata -> %s) @> %s::jsonb)",
            [key, key, _jsonb([value])],
        )
    if isinstance(value, int):
        elem = "(jsonb_typeof(e) = 'number' AND e::text !~ '[.eE]' AND e::text::numeric = %s::numeric)"
    else:
        elem = "(jsonb_typeof(e) = 'number' AND e::text ~ '[.eE]' AND e::text::float8 = %s::float8)"
    return (
        "EXISTS (SELECT 1 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(metadata -> %s) = "
        f"'array' THEN metadata -> %s ELSE '[]'::jsonb END) AS e WHERE {elem})",
        [key, key, value],
    )


def _negate(expr: tuple[str, list]) -> tuple[str, list]:
    sql, params = expr
    return f"(NOT COALESCE({sql}, FALSE))", params


def _field(key: str, cond: Any) -> tuple[str, list]:
    if _is_scalar(cond):
        return _eq(key, cond)
    if not isinstance(cond, dict):
        raise InvalidArgumentError(
            f"Expected where value to be a str, int, float, or operator expression, got {cond!r}"
        )
    if len(cond) != 1:
        raise InvalidArgumentError(
            f"Expected operator expression to have exactly one operator, got {cond!r}"
        )
    op, value = next(iter(cond.items()))
    if op not in _WHERE_OPS:
        raise InvalidArgumentError(
            f"Expected where operator to be one of {', '.join(_WHERE_OPS)}, got {op}"
        )
    if op in _COMPARE:
        if not _is_number(value):
            raise InvalidArgumentError(
                f"Expected operand value to be an int or a float for operator {op}, got {value!r}"
            )
        sym = _COMPARE[op]
        return _num_case(
            key,
            f"(metadata ->> %s)::float8 {sym} %s::float8",
            [key, float(value)],
            f"(metadata ->> %s)::numeric {sym} %s::numeric",
            [key, _int_operand(value)],
        )
    if op in ("$eq", "$ne"):
        if not _is_scalar(value):
            raise InvalidArgumentError(
                f"Expected operand value to be a str, int, float, or bool for operator {op}, got {value!r}"
            )
        expr = _eq(key, value)
        return expr if op == "$eq" else _negate(expr)
    if op in ("$in", "$nin"):
        if (
            not isinstance(value, list)
            or not value
            or not all(_is_scalar(v) for v in value)
            or len({str if isinstance(v, str) else bool if isinstance(v, bool) else float for v in value}) != 1
        ):
            raise InvalidArgumentError(
                f"Expected where operand value to be a non-empty list, and all values to be of the same type got {value!r}"
            )
        expr = _in(key, value)
        return expr if op == "$in" else _negate(expr)
    # $contains / $not_contains on a list-valued metadata key
    if not _is_scalar(value):
        raise InvalidArgumentError(
            f"Expected operand value to be a str, int, float, or bool for operator {op}, got {value!r}"
        )
    expr = _list_contains(key, value)
    return expr if op == "$contains" else _negate(expr)


def _logical(op: str, clauses: Any, compile_one) -> tuple[str, list]:
    if not isinstance(clauses, list) or not clauses:
        raise InvalidArgumentError(f"Expected {op} to be a non-empty list, got {clauses!r}")
    parts = [compile_one(c) for c in clauses]
    joiner = " AND " if op == "$and" else " OR "
    params: list = []
    for _, p in parts:
        params.extend(p)
    return "(" + joiner.join(f"COALESCE({s}, FALSE)" for s, _ in parts) + ")", params


def _where(where: Any) -> tuple[str, list]:
    if not isinstance(where, dict) or not where:
        raise InvalidArgumentError(f"Expected where to be a non-empty dict, got {where!r}")
    parts = []
    for key, cond in where.items():
        if key in ("$and", "$or"):
            parts.append(_logical(key, cond, _where))
        elif not isinstance(key, str) or key.startswith("$"):
            raise InvalidArgumentError(f"Unknown where operator {key!r}")
        else:
            parts.append(_field(key, cond))
    if len(parts) == 1:
        return parts[0]
    params: list = []
    for _, p in parts:
        params.extend(p)
    return "(" + " AND ".join(f"COALESCE({s}, FALSE)" for s, _ in parts) + ")", params


def _document(wd: Any) -> tuple[str, list]:
    if not isinstance(wd, dict) or len(wd) != 1:
        raise InvalidArgumentError(
            f"Expected where document to have exactly one operator, got {wd!r}"
        )
    op, value = next(iter(wd.items()))
    if op in ("$and", "$or"):
        return _logical(op, value, _document)
    if op not in _DOC_OPS:
        raise InvalidArgumentError(
            f"Expected where document operator to be one of {', '.join(_DOC_OPS + ('$and', '$or'))}, got {op}"
        )
    if not isinstance(value, str) or not value:
        raise InvalidArgumentError(
            f"Expected where document operand value for operator {op} to be a non-empty str"
        )
    if op == "$contains":
        return "(strpos(document, %s) > 0)", [value]
    if op == "$not_contains":
        return _negate(("(strpos(document, %s) > 0)", [value]))
    if op == "$regex":
        return "(document ~ %s)", [value]
    return _negate(("(document ~ %s)", [value]))


def compile_where(where: dict | None) -> tuple[str, list]:
    if where is None or where == {}:
        return "TRUE", []
    return _where(where)


def compile_where_document(where_document: dict | None) -> tuple[str, list]:
    if where_document is None or where_document == {}:
        return "TRUE", []
    return _document(where_document)
