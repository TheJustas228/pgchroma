import numpy as np
import pytest

from pgchroma.store import Store

T, D = "default_tenant", "default_database"


@pytest.fixture
async def store(pool):
    s = Store(pool, pool.pgchroma_info, index_threshold=50)
    yield s
    await s.wait_for_indexing()


async def make(store, name="docs", space="l2"):
    meta = await store.create_collection(T, D, name, None, {"hnsw": {"space": space}}, None, False)
    return await store.collection(T, D, meta["id"])


async def hnsw_indexes(store, coll):
    async with store.pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = %s AND indexdef ILIKE '%%hnsw%%'",
                (f"c_{coll.id.hex}",),
            )
        ).fetchall()
    return [r[0] for r in rows]


async def test_index_built_once_threshold_reached(store):
    c = await make(store, space="cosine")
    vecs = np.random.default_rng(1).random((40, 8), dtype=np.float32)
    await store.add(c, [f"a{i}" for i in range(40)], vecs, None, None, None)
    await store.wait_for_indexing()
    assert await hnsw_indexes(store, c) == []
    await store.add(c, [f"b{i}" for i in range(20)], vecs[:20], None, None, None)
    await store.wait_for_indexing()
    defs = await hnsw_indexes(store, c)
    assert len(defs) == 1 and "vector_cosine_ops" in defs[0]


async def test_filtered_query_on_indexed_collection_returns_enough_results(store):
    c = await make(store)
    rng = np.random.default_rng(2)
    n = 400
    vecs = rng.random((n, 16), dtype=np.float32)
    metas = [{"bucket": "rare" if i % 20 == 0 else "common"} for i in range(n)]
    await store.add(c, [f"r{i}" for i in range(n)], vecs, metas, None, None)
    await store.wait_for_indexing()
    assert await hnsw_indexes(store, c)
    r = await store.query(
        c, vecs[:1], 10, None, {"bucket": "rare"}, None, ["metadatas", "distances"]
    )
    assert len(r["ids"][0]) == 10
    assert all(m["bucket"] == "rare" for m in r["metadatas"][0])
    assert r["distances"][0] == sorted(r["distances"][0])


async def test_indexed_results_match_exact_search(store):
    c = await make(store)
    rng = np.random.default_rng(3)
    vecs = rng.random((300, 32), dtype=np.float32)
    await store.add(c, [f"v{i}" for i in range(300)], vecs, None, None, None)
    await store.wait_for_indexing()
    queries = vecs[:20]
    r = await store.query(c, queries, 5, None, None, None, [])
    exact = np.argsort(((vecs[None, :, :] - queries[:, None, :]) ** 2).sum(-1), axis=1)[:, :5]
    recall = np.mean(
        [len(set(r["ids"][i]) & {f"v{j}" for j in exact[i]}) / 5 for i in range(len(queries))]
    )
    assert recall >= 0.95


async def test_high_dimensional_collection(store):
    c = await make(store, space="cosine")
    vecs = np.random.default_rng(4).random((60, 3072), dtype=np.float32)
    await store.add(c, [f"h{i}" for i in range(60)], vecs, None, None, None)
    await store.wait_for_indexing()
    defs = await hnsw_indexes(store, c)
    if store.info.has_halfvec:
        assert len(defs) == 1 and "halfvec" in defs[0]
    else:
        assert defs == []  # exact search only on pgvector < 0.7
    r = await store.query(c, vecs[5:6], 3, None, None, None, ["distances"])
    assert r["ids"][0][0] == "h5" and r["distances"][0][0] < 1e-3
