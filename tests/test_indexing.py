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


async def test_index_is_built_when_reads_need_it_not_on_writes(store):
    c = await make(store, space="cosine")
    vecs = np.random.default_rng(1).random((60, 8), dtype=np.float32)
    await store.add(c, [f"a{i}" for i in range(40)], vecs[:40], None, None, None)
    await store.query(c, vecs[:1], 3, None, None, None, [])
    await store.wait_for_indexing()
    assert await hnsw_indexes(store, c) == []  # below the threshold: exact search
    await store.add(c, [f"b{i}" for i in range(20)], vecs[40:], None, None, None)
    await store.wait_for_indexing()
    assert await hnsw_indexes(store, c) == []  # writes alone never pay for index maintenance
    await store.query(c, vecs[:1], 3, None, None, None, [])
    await store.wait_for_indexing()
    defs = await hnsw_indexes(store, c)
    assert len(defs) == 1 and "vector_cosine_ops" in defs[0]


async def test_build_indexes_explicitly(store):
    c = await make(store)
    vecs = np.random.default_rng(5).random((80, 8), dtype=np.float32)
    await store.add(c, [f"x{i}" for i in range(80)], vecs, None, None, None)
    assert await store.indexing_progress(c) == 0.0
    assert await store.build_index(c) is True
    assert await hnsw_indexes(store, c)
    assert await store.indexing_progress(c) == 1.0


async def test_small_collection_reports_fully_indexed(store):
    c = await make(store)
    await store.add(c, ["one"], np.ones((1, 4), dtype=np.float32), None, None, None)
    assert await store.indexing_progress(c) == 1.0


async def test_filtered_query_on_indexed_collection_returns_enough_results(store):
    c = await make(store)
    rng = np.random.default_rng(2)
    n = 400
    vecs = rng.random((n, 16), dtype=np.float32)
    metas = [{"bucket": "rare" if i % 20 == 0 else "common"} for i in range(n)]
    await store.add(c, [f"r{i}" for i in range(n)], vecs, metas, None, None)
    await store.build_index(c)
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
    await store.build_index(c)
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
    await store.build_index(c)
    defs = await hnsw_indexes(store, c)
    if store.info.has_halfvec:
        assert len(defs) == 1 and "halfvec" in defs[0]
    else:
        assert defs == []  # exact search only on pgvector < 0.7
    r = await store.query(c, vecs[5:6], 3, None, None, None, ["distances"])
    assert r["ids"][0][0] == "h5" and r["distances"][0][0] < 1e-3


async def test_knn_query_plan_uses_the_hnsw_index(store):
    c = await make(store, space="cosine")
    vecs = np.random.default_rng(6).random((500, 16), dtype=np.float32)
    await store.add(c, [f"p{i}" for i in range(500)], vecs, None, None, None)
    await store.build_index(c)
    async with store.pool.connection() as conn:
        await conn.execute(f"ANALYZE c_{c.id.hex}")
        await conn.execute("SET enable_seqscan = off")  # make the cheapest-plan choice unambiguous
        captured = []
        original = conn.execute

        async def explain(query, params=None, **kw):
            text = query.as_string(conn) if hasattr(query, "as_string") else query
            if "CROSS JOIN LATERAL" in text:
                captured.append(await (await original("EXPLAIN " + text, params, **kw)).fetchall())
            return await original(query, params, **kw)

        conn.execute = explain
        store.pool.connection = lambda *a, **k: _Fixed(conn)
        await store.query(c, vecs[:1], 5, None, None, None, [])
    plan = "\n".join(r[0] for r in captured[0])
    assert "_hnsw" in plan, plan


class _Fixed:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


async def test_large_n_results_on_indexed_collection(store):
    c = await make(store)
    vecs = np.random.default_rng(7).random((1600, 8), dtype=np.float32)
    await store.add(c, [f"n{i}" for i in range(1600)], vecs, None, None, None)
    await store.build_index(c)
    r = await store.query(c, vecs[:1], 1500, None, None, None, ["distances"])
    assert len(r["ids"][0]) == 1500
    assert r["distances"][0] == sorted(r["distances"][0])


async def test_fork_of_indexed_collection_gets_exactly_one_hnsw_index(store):
    c = await make(store)
    vecs = np.random.default_rng(8).random((120, 8), dtype=np.float32)
    await store.add(c, [f"f{i}" for i in range(120)], vecs, None, None, None)
    await store.build_index(c)
    forked = await store.fork_collection(
        "default_tenant", "default_database", str(c.id), "docs-fork"
    )
    fc = await store.collection("default_tenant", "default_database", forked["id"])
    await store.build_index(fc)
    assert len(await hnsw_indexes(store, fc)) == 1
    r = await store.query(fc, vecs[:1], 3, None, None, None, [])
    assert r["ids"][0][0] == "f0"
