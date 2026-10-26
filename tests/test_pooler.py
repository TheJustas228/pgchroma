"""pgchroma behind a transaction pooler (PgBouncer, Supavisor, Neon's pooled endpoint).

A transaction pooler runs each transaction on whichever server connection is free, so session
state (SET search_path, session advisory locks, session settings) does not carry over from one
statement to the next. These tests take that state away and check that everything still works.
Set PGCHROMA_TEST_POOLER_URL to also run them through a real transaction pooler.
"""

import contextlib
import os
import uuid

import numpy as np
import pytest
from psycopg import sql

from pgchroma.db import bootstrap, detect_transaction_pooler, open_pool
from pgchroma.errors import NotFoundError
from pgchroma.indexing import ensure_index, index_plan
from pgchroma.search import search
from pgchroma.store import Store

T, D = "default_tenant", "default_database"


@contextlib.asynccontextmanager
async def lost_session_pool(url, size=3):
    """A pool whose connections forgot pgchroma's search_path, as behind a transaction pooler."""
    schema = f"t_{uuid.uuid4().hex[:12]}"
    pool = await open_pool(url, schema, min_size=size, max_size=size)
    try:
        await bootstrap(pool, schema)
        async with contextlib.AsyncExitStack() as stack:
            conns = [await stack.enter_async_context(pool.connection()) for _ in range(size)]
            for conn in conns:
                await conn.execute("SET search_path TO public")
                await conn.commit()
        yield pool
    finally:
        async with pool.connection() as conn:
            await conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
        await pool.close()


async def exercise(pool):
    store = Store(pool, pool.pgchroma_info, index_threshold=100)
    meta = await store.create_collection(
        T, D, "pooled", None, {"hnsw": {"space": "cosine"}}, None, False
    )
    ref = await store.collection(T, D, meta["id"])
    vecs = np.random.default_rng(3).normal(size=(150, 8)).astype(np.float32)
    await store.add(
        ref, [f"r{i}" for i in range(150)], vecs, [{"n": i % 5} for i in range(150)], None, None
    )
    ref = await store.collection(T, D, meta["id"])
    assert await store.count(ref) == 150
    got = await store.get(ref, None, {"n": 2}, None, None, None, ["metadatas"])
    assert len(got["ids"]) == 30
    assert await store.build_index(ref) is True
    assert await store.indexing_progress(ref) == 1.0
    hits = await store.query(ref, vecs[:2], 5, None, {"n": 0}, None, ["distances"])
    assert [len(h) for h in hits["ids"]] == [5, 5]
    found = await search(
        store, ref, {"searches": [{"rank": {"$knn": {"query": vecs[0].tolist(), "limit": 3}}}]}
    )
    assert found["ids"][0][0] == "r0"
    await store.delete_collection(T, D, "pooled")


async def test_works_when_connections_lose_session_state(database_url, monkeypatch):
    monkeypatch.setenv("PGCHROMA_POOLER", "transaction")
    async with lost_session_pool(database_url) as pool:
        await exercise(pool)


async def test_switches_to_pooler_mode_when_the_search_path_is_lost(database_url, monkeypatch):
    monkeypatch.delenv("PGCHROMA_POOLER", raising=False)
    async with lost_session_pool(database_url) as pool:
        assert pool.pgchroma_pooled is False
        store = Store(pool, pool.pgchroma_info)
        with pytest.raises(NotFoundError):
            await store.list_collections(T, D, None, None)
        assert pool.pgchroma_pooled is True
        await exercise(pool)


async def test_index_build_lease_blocks_a_second_builder_but_not_a_stale_one(pool):
    store = Store(pool, pool.pgchroma_info)
    meta = await store.create_collection(
        T, D, "leased", None, {"hnsw": {"space": "l2"}}, None, False
    )
    ref = await store.collection(T, D, meta["id"])
    await store.add(ref, ["a", "b"], np.eye(2, 4, dtype=np.float32), None, None, None)
    ref = await store.collection(T, D, meta["id"])
    table = f"c_{ref.id.hex}"
    plan = index_plan(4, "l2", pool.pgchroma_info)
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO index_builds (table_name, owner) VALUES (%s, 'someone-else')", (table,)
        )
    assert await ensure_index(pool, table, plan, {}, 0) is False  # a live lease: skip
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE index_builds SET started_at = now() - interval '1 hour' WHERE table_name = %s",
            (table,),
        )
    assert await ensure_index(pool, table, plan, {}, 0) is True  # stale and nothing building
    async with pool.connection() as conn:
        left = await (await conn.execute("SELECT count(*) FROM index_builds")).fetchone()
    assert left[0] == 0


@pytest.mark.parametrize(
    "url,pooled",
    [
        ("postgresql://u:p@ep-cool-name-123-pooler.us-east-2.aws.neon.tech/db", True),
        ("postgresql://u:p@ep-cool-name-123.us-east-2.aws.neon.tech/db", False),
        ("postgresql://u:p@aws-0-eu-central-1.pooler.supabase.com:6543/postgres", True),
        ("postgresql://u:p@aws-0-eu-central-1.pooler.supabase.com:5432/postgres", False),
        ("postgresql://u:p@db.abcd.supabase.co:5432/postgres", False),
        ("host=localhost port=5432 dbname=app", False),
    ],
)
def test_detects_transaction_poolers_from_the_url(url, pooled, monkeypatch):
    monkeypatch.delenv("PGCHROMA_POOLER", raising=False)
    assert detect_transaction_pooler(url) is pooled


def test_pooler_can_be_declared(monkeypatch):
    monkeypatch.setenv("PGCHROMA_POOLER", "transaction")
    assert detect_transaction_pooler("postgresql://localhost/app") is True


@pytest.mark.skipif(not os.environ.get("PGCHROMA_TEST_POOLER_URL"), reason="no pooler configured")
async def test_through_a_real_transaction_pooler(monkeypatch):
    monkeypatch.setenv("PGCHROMA_POOLER", "transaction")
    async with lost_session_pool(os.environ["PGCHROMA_TEST_POOLER_URL"]) as pool:
        await exercise(pool)
