import asyncio
import uuid

import numpy as np
import pytest

from pgchroma.db import bootstrap, open_pool
from pgchroma.errors import DuplicateError, InvalidArgumentError, NotFoundError
from pgchroma.store import Store
from pgchroma.vectors import decode_embeddings

T, D = "default_tenant", "default_database"
ALL = ["embeddings", "documents", "metadatas", "uris"]


@pytest.fixture
async def store(pool):
    return Store(pool, pool.pgchroma_info)


async def make(store, name="docs", **kw):
    meta = await store.create_collection(
        T, D, name, kw.get("metadata"), kw.get("configuration"), None, False
    )
    return await store.collection(T, D, meta["id"])


def emb(rows):
    return np.asarray(rows, dtype=np.float32)


async def test_collection_json_shape(store):
    c = await store.create_collection(
        T, D, "docs", {"topic": "t"}, {"hnsw": {"space": "cosine"}}, None, False
    )
    assert set(c) == {
        "id",
        "name",
        "configuration_json",
        "schema",
        "metadata",
        "dimension",
        "tenant",
        "database",
        "log_position",
        "version",
    }
    assert c["configuration_json"]["hnsw"] == {
        "space": "cosine",
        "ef_construction": 100,
        "ef_search": 100,
        "max_neighbors": 16,
        "resize_factor": 1.2,
        "sync_threshold": 1000,
    }
    assert c["configuration_json"]["embedding_function"] is None
    assert c["schema"]["defaults"]["float_list"]["vector_index"]["config"]["space"] == "cosine"
    assert (c["metadata"], c["dimension"], c["tenant"], c["database"]) == (
        {"topic": "t"},
        None,
        T,
        D,
    )


async def test_duplicate_and_missing_collections(store):
    await store.create_collection(T, D, "docs", None, None, None, False)
    with pytest.raises(DuplicateError, match=r"Collection \[docs\] already exists"):
        await store.create_collection(T, D, "docs", None, None, None, False)
    again = await store.create_collection(T, D, "docs", {"ignored": 1}, None, None, True)
    assert again["metadata"] is None
    with pytest.raises(NotFoundError, match=r"Collection \[nope\] does not exist"):
        await store.get_collection(T, D, "nope")


@pytest.mark.parametrize("name", ["f", "-abc", "abc-", "a b c", "x" * 513])
async def test_invalid_collection_names(store, name):
    with pytest.raises(InvalidArgumentError, match="Expected a name containing 3-512 characters"):
        await store.create_collection(T, D, name, None, None, None, False)


async def test_add_get_roundtrip_keeps_types_and_order(store):
    c = await make(store)
    metas = [{"n": 1, "s": "x", "f": 1.5, "b": True, "l": [1, 2]}, {"n": 2.0}, None]
    await store.add(
        c,
        ["a", "b", "c"],
        emb([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        metas,
        ["A", "B", None],
        [None, "u", None],
    )
    r = await store.get(c, None, None, None, None, None, ALL)
    assert r["ids"] == ["a", "b", "c"]
    assert r["metadatas"] == metas
    assert isinstance(r["metadatas"][0]["n"], int) and isinstance(r["metadatas"][1]["n"], float)
    assert r["documents"] == ["A", "B", None]
    assert r["uris"] == [None, "u", None]
    assert np.allclose(np.asarray(r["embeddings"]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c = await store.collection(T, D, c.id)
    assert c.dimension == 3


async def test_add_existing_id_is_ignored(store):
    c = await make(store)
    await store.add(c, ["a"], emb([[1, 0]]), None, ["first"], None)
    await store.add(c, ["a"], emb([[9, 9]]), None, ["second"], None)
    r = await store.get(c, None, None, None, None, None, ["documents"])
    assert r["documents"] == ["first"]


async def test_duplicate_ids_in_batch_rejected(store):
    c = await make(store)
    with pytest.raises(InvalidArgumentError, match="unique"):
        await store.add(c, ["a", "a"], emb([[1, 0], [0, 1]]), None, None, None)


async def test_dimension_mismatch(store):
    c = await make(store)
    await store.add(c, ["a"], emb([[1, 0, 0]]), None, None, None)
    with pytest.raises(
        InvalidArgumentError, match="Collection expecting embedding with dimension of 3, got 2"
    ):
        await store.add(c, ["b"], emb([[1, 0]]), None, None, None)
    with pytest.raises(InvalidArgumentError, match="dimension of 3, got 2"):
        await store.query(c, emb([[1, 0]]), 1, None, None, None, ["distances"])


async def test_update_merges_metadata_and_ignores_missing(store):
    c = await make(store)
    await store.add(
        c, ["b", "c"], emb([[1, 0], [0, 1]]), [{"n": 2, "s": "y"}, {"z": 1}], ["B", "C"], None
    )
    await store.update(
        c, ["b", "c", "missing"], None, [{"n": 5, "s": None}, {"z": None}, {"q": 1}], None, None
    )
    r = await store.get(c, None, None, None, None, None, ["metadatas", "documents"])
    assert r["metadatas"] == [{"n": 5}, {}]
    assert r["documents"] == ["B", "C"]


async def test_upsert_merges_and_keeps_position(store):
    c = await make(store)
    await store.add(
        c,
        ["a", "b"],
        emb([[1, 0], [0, 1]]),
        [{"n": 2, "s": "y"}, {"k": 1}],
        ["A", "B"],
        ["u1", "u2"],
    )
    await store.upsert(
        c, ["a", "d"], emb([[2, 2], [3, 3]]), [{"x": 1, "s": None}, {"d": 1}], None, None
    )
    r = await store.get(c, None, None, None, None, None, ALL)
    assert r["ids"] == ["a", "b", "d"]
    assert r["metadatas"] == [{"n": 2, "x": 1}, {"k": 1}, {"d": 1}]
    assert r["documents"] == ["A", "B", None]
    assert r["uris"] == ["u1", "u2", None]
    assert np.allclose(np.asarray(r["embeddings"][0]), [2, 2])


async def test_get_filters_ids_limit_offset(store):
    c = await make(store)
    await store.add(
        c, list("abcde"), emb([[i, 0] for i in range(5)]), [{"i": i} for i in range(5)], None, None
    )
    r = await store.get(c, ["e", "a", "zz"], None, None, None, None, [])
    assert r["ids"] == ["a", "e"]
    r = await store.get(c, None, {"i": {"$gte": 1}}, None, 2, 1, [])
    assert r["ids"] == ["c", "d"]
    assert r["metadatas"] is None and r["embeddings"] is None


async def test_query_distances_per_space(store):
    q = emb([[1, 0.1, 0]])
    rows = emb([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for space, expected in {
        "l2": [0.01, 1.81, 2.01],
        "cosine": [1 - 1 / np.sqrt(1.01), 1 - 0.1 / np.sqrt(1.01), 1.0],
        "ip": [0.0, 0.9, 1.0],
    }.items():
        c = await make(store, f"col-{space}", configuration={"hnsw": {"space": space}})
        await store.add(c, ["a", "b", "c"], rows, None, None, None)
        r = await store.query(c, q, 10, None, None, None, ["distances"])
        assert r["ids"] == [["a", "b", "c"]]
        assert np.allclose(r["distances"][0], expected, atol=1e-5), space


async def test_query_filters_and_empty(store):
    c = await make(store)
    r = await store.query(c, emb([[1, 0]]), 3, None, None, None, ["distances", "documents"])
    assert r["ids"] == [[]] and r["distances"] == [[]] and r["documents"] == [[]]
    await store.add(
        c,
        ["a", "b", "c"],
        emb([[1, 0], [0, 1], [1, 1]]),
        [{"k": 1}, {"k": 2}, {"k": 2}],
        ["x", "y", "z"],
        None,
    )
    r = await store.query(c, emb([[1, 0], [0, 1]]), 5, None, {"k": 2}, None, ["documents"])
    assert r["ids"] == [["c", "b"], ["b", "c"]]
    r = await store.query(c, emb([[1, 0]]), 5, ["a", "b", "nope"], None, None, [])
    assert r["ids"] == [["a", "b"]]


async def test_delete_and_count(store):
    c = await make(store)
    await store.add(
        c,
        list("abc"),
        emb([[1, 0], [0, 1], [1, 1]]),
        [{"s": "x"}, {"s": "y"}, None],
        ["a2", "b2", "c"],
        None,
    )
    assert await store.delete(c, ["a", "b", "c"], None, {"$contains": "2"}, None) == 2
    assert await store.count(c) == 1
    assert await store.delete(c, None, {"s": "nope"}, None, None) == 0
    # pgchroma reports the exact number deleted (Chroma reports len(ids) for id-only deletes)
    assert await store.delete(c, ["c", "missing"], None, None, None) == 1


async def test_collections_lifecycle(store):
    a = await store.create_collection(T, D, "aaa", None, None, None, False)
    await store.create_collection(T, D, "bbb", None, None, None, False)
    listed = await store.list_collections(T, D, None, None)
    assert [c["id"] for c in listed] == sorted(c["id"] for c in listed)  # Chroma orders by id
    assert {c["name"] for c in listed} == {"aaa", "bbb"}
    assert [c["id"] for c in await store.list_collections(T, D, 1, 1)] == [listed[1]["id"]]
    assert await store.count_collections(T, D) == 2
    await store.update_collection(T, D, a["id"], "ccc", {"m": 1}, None)
    got = await store.get_collection(T, D, "ccc")
    assert (got["id"], got["metadata"]) == (a["id"], {"m": 1})
    await store.delete_collection(T, D, "ccc")
    with pytest.raises(NotFoundError):
        await store.delete_collection(T, D, "ccc")
    assert await store.count_collections(T, D) == 1


async def test_tenants_and_databases(store):
    await store.create_tenant("acme")
    assert await store.get_tenant("acme") == {"name": "acme", "resource_name": None}
    with pytest.raises(DuplicateError):
        await store.create_tenant("acme")
    await store.create_database("acme", "prod")
    db = await store.get_database("acme", "prod")
    assert db["name"] == "prod" and db["tenant"] == "acme" and uuid.UUID(db["id"])
    assert [d["name"] for d in await store.list_databases("acme", None, None)] == ["prod"]
    c = await store.create_collection("acme", "prod", "things", None, None, None, False)
    await store.delete_database("acme", "prod")
    with pytest.raises(NotFoundError):
        await store.get_database("acme", "prod")
    async with store.pool.connection() as conn:
        row = await (
            await conn.execute("SELECT to_regclass(%s)", (f"c_{uuid.UUID(c['id']).hex}",))
        ).fetchone()
    assert row[0] is None, "collection table should be dropped with its database"
    with pytest.raises(NotFoundError):
        await store.get_tenant("nope")


async def test_fork_copies_records(store):
    c = await make(store)
    await store.add(c, ["a", "b"], emb([[1, 0], [0, 1]]), [{"k": 1}, None], ["A", "B"], None)
    f = await store.fork_collection(T, D, str(c.id), "docs-fork")
    fc = await store.collection(T, D, f["id"])
    r = await store.get(fc, None, None, None, None, None, ["documents", "metadatas"])
    assert r["ids"] == ["a", "b"] and r["documents"] == ["A", "B"]
    await store.add(fc, ["z"], emb([[1, 1]]), None, None, None)
    assert await store.count(c) == 2


async def test_add_max_batch(store):
    c = await make(store)
    n, dim = 5461, 384
    vecs = np.random.default_rng(0).random((n, dim), dtype=np.float32)
    ids = [f"id-{i}" for i in range(n)]
    await store.add(
        c, ids, vecs, [{"i": i} for i in range(n)], [f"doc {i}" for i in range(n)], None
    )
    assert await store.count(c) == n
    r = await store.query(c, vecs[:1], 1, None, None, None, ["distances"])
    assert r["ids"] == [["id-0"]] and r["distances"][0][0] < 1e-6


async def test_concurrent_first_write(database_url, pool):
    schema = pool.pgchroma_schema
    other = await open_pool(database_url, schema)
    try:
        s1, s2 = (
            Store(pool, await bootstrap(pool, schema)),
            Store(other, await bootstrap(other, schema)),
        )
        c = await make(s1)
        c2 = await s2.collection(T, D, c.id)
        await asyncio.gather(
            *[
                s.add(cc, [f"{tag}{i}"], emb([[1, 2, 3]]), None, None, None)
                for i in range(10)
                for s, cc, tag in ((s1, c, "x"), (s2, c2, "y"))
            ]
        )
        assert await s1.count(c) == 20
    finally:
        await other.close()


def test_decode_embeddings_accepts_base64_and_lists():
    import base64

    raw = base64.b64encode(np.asarray([1.0, 0.5], dtype="<f4").tobytes()).decode()
    assert decode_embeddings([raw, [3, 4]]).tolist() == [[1.0, 0.5], [3.0, 4.0]]
    assert decode_embeddings(None) is None
    with pytest.raises(InvalidArgumentError):
        decode_embeddings([[1, 2], [1, 2, 3]])


async def test_collection_cache_is_invalidated_by_local_changes(store):
    c = await make(store)
    await store.add(c, ["a"], emb([[1, 0]]), None, None, None)
    ref1 = await store.collection(T, D, c.id)
    ref2 = await store.collection(T, D, c.id)
    assert ref1 is ref2  # served from the short-lived cache
    await store.update_collection(T, D, c.id, "renamed", {"m": 1}, None)
    assert (await store.collection(T, D, c.id)).name == "renamed"
    await store.delete_collection(T, D, "renamed")
    with pytest.raises(NotFoundError):
        await store.collection(T, D, c.id)


async def test_collections_without_dimension_are_never_cached(store):
    c = await make(store)
    assert (await store.collection(T, D, c.id)) is not (await store.collection(T, D, c.id))
