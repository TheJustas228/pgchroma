"""Filter semantics. Expected ids were recorded from a real Chroma 1.5.9 server on these rows."""

import orjson
import pytest

from pgchroma.errors import InvalidArgumentError
from pgchroma.filters import compile_where, compile_where_document

ROWS = [
    ("int1", {"n": 1, "tag": "a"}, "Hello World"),
    ("float1", {"n": 1.0}, "100% sure_thing"),
    ("float15", {"n": 1.5}, None),
    ("bool_t", {"n": True}, None),
    ("str1", {"n": "1"}, "back\\slash"),
    ("none", None, None),
    ("list", {"l": [1, 2, 3], "ls": ["x", "y"]}, "unicode ü 日本"),
    ("listf", {"l": [1.5]}, None),
    ("big", {"n": 10}, None),
    ("neg", {"n": -2}, None),
    ("strs", {"s": "apple"}, 'it\'s "quoted"'),
    ("strs2", {"s": "Banana"}, None),
]
ALL = [r[0] for r in ROWS]


def without(*ids):
    return [i for i in ALL if i not in ids]


@pytest.fixture
async def conn(pool):
    async with pool.connection() as conn:
        await conn.execute(
            "CREATE TEMP TABLE t (seq serial, id text, metadata jsonb, document text)"
        )
        for rid, meta, doc in ROWS:
            await conn.execute(
                "INSERT INTO t (id, metadata, document) VALUES (%s, %s::jsonb, %s)",
                (rid, None if meta is None else orjson.dumps(meta).decode(), doc),
            )
        yield conn


async def select_ids(conn, where=None, where_document=None):
    w_sql, w_params = compile_where(where)
    d_sql, d_params = compile_where_document(where_document)
    cur = await conn.execute(
        f"SELECT id FROM t WHERE ({w_sql}) AND ({d_sql}) ORDER BY seq", w_params + d_params
    )
    return [r[0] for r in await cur.fetchall()]


WHERE_CASES = [
    ({"n": 1}, ["int1", "float1"]),
    ({"n": 1.0}, ["int1", "float1"]),
    ({"n": True}, ["bool_t"]),
    ({"n": "1"}, ["str1"]),
    ({"n": {"$eq": 1}}, ["int1", "float1"]),
    ({"n": {"$ne": 1}}, without("int1", "float1")),
    ({"n": {"$ne": True}}, without("bool_t")),
    ({"n": {"$gt": 1}}, ["float15", "big"]),
    ({"n": {"$gte": 1}}, ["int1", "float1", "float15", "big"]),
    ({"n": {"$lt": 1.5}}, ["float1", "neg"]),
    ({"n": {"$in": [1]}}, ["int1", "float1"]),
    ({"n": {"$in": [1.0]}}, ["int1", "float1"]),
    ({"n": {"$nin": [1]}}, without("int1", "float1")),
    ({"n": {"$in": [True]}}, ["bool_t"]),
    ({"l": {"$contains": 2}}, ["list"]),
    ({"l": {"$contains": 1.5}}, ["listf"]),
    ({"ls": {"$contains": "x"}}, ["list"]),
    ({"l": {"$not_contains": 2}}, without("list")),
    ({"l": 2}, []),
    ({"$and": [{"n": {"$gte": 1}}, {"n": {"$lte": 1.5}}]}, ["int1", "float1", "float15"]),
    ({"$or": [{"tag": "a"}, {"s": "apple"}]}, ["int1", "strs"]),
    # Chroma casts a float operand to an integer (toward zero) when it meets integer metadata.
    ({"n": 1.5}, ["int1", "float15"]),
    ({"n": 1.9}, ["int1"]),
    ({"n": {"$gt": -2.5}}, ["int1", "float1", "float15", "big"]),
    ({"n": {"$gte": -2.5}}, ["int1", "float1", "float15", "big", "neg"]),
    ({"n": {"$lt": -1.5}}, ["neg"]),
    ({"n": {"$lte": 0.9}}, ["neg"]),
    ({"n": {"$ne": 1.5}}, without("int1", "float15")),
    ({"n": {"$in": [1.5]}}, ["int1", "float15"]),
    ({"n": {"$nin": [1.5]}}, without("int1", "float15")),
    ({"n": {"$gt": 9.99}}, ["big"]),
    ({"l": {"$contains": 1.0}}, []),
    ({"l": {"$contains": 2.7}}, []),
]


@pytest.mark.parametrize("where,expected", WHERE_CASES, ids=[str(c[0]) for c in WHERE_CASES])
async def test_where(conn, where, expected):
    assert await select_ids(conn, where=where) == expected


DOC_CASES = [
    ({"$contains": "World"}, ["int1"]),
    ({"$contains": "world"}, []),
    ({"$contains": "%"}, ["float1"]),
    ({"$contains": "_"}, ["float1"]),
    ({"$contains": "\\"}, ["str1"]),
    ({"$contains": '"'}, ["strs"]),
    ({"$contains": "日本"}, ["list"]),
    ({"$not_contains": "o"}, without("int1", "list", "strs")),
    ({"$regex": "^H.*d$"}, ["int1"]),
    ({"$regex": "(?i)hello"}, ["int1"]),
    ({"$not_regex": "o"}, without("int1", "list", "strs")),
    ({"$and": [{"$contains": "o"}, {"$not_contains": "W"}]}, ["list", "strs"]),
    ({"$or": [{"$contains": "World"}, {"$contains": "sure"}]}, ["int1", "float1"]),
]


@pytest.mark.parametrize("where_document,expected", DOC_CASES, ids=[str(c[0]) for c in DOC_CASES])
async def test_where_document(conn, where_document, expected):
    assert await select_ids(conn, where_document=where_document) == expected


async def test_where_and_document_combine(conn):
    assert await select_ids(conn, where={"n": {"$gte": 1}}, where_document={"$contains": "o"}) == [
        "int1"
    ]


def test_none_compiles_to_true():
    assert compile_where(None) == ("TRUE", [])
    assert compile_where_document(None) == ("TRUE", [])


@pytest.mark.parametrize(
    "where",
    [
        {"n": {"$foo": 1}},
        {"n": {"$in": []}},
        {"n": {"$gt": "b"}},
        {"n": {"$gt": True}},
        {"n": [1, 2]},
        {"$and": []},
        {"$and": {"n": 1}},
        {"n": {"$eq": 1, "$ne": 2}},
        {"n": None},
    ],
)
def test_invalid_where_raises(where):
    with pytest.raises(InvalidArgumentError):
        compile_where(where)


@pytest.mark.parametrize(
    "where_document",
    [{"$contains": ""}, {"$contains": 3}, {"$foo": "x"}, {"$or": []}],
)
def test_invalid_where_document_raises(where_document):
    with pytest.raises(InvalidArgumentError):
        compile_where_document(where_document)


def test_empty_filters_mean_no_filter():
    assert compile_where({}) == ("TRUE", [])
    assert compile_where_document({}) == ("TRUE", [])
