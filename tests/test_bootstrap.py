from pgchroma.db import bootstrap

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


async def test_bootstrap_is_idempotent(pool):
    schema = pool.pgchroma_schema
    info = await bootstrap(pool, schema)  # second run on top of the fixture's
    assert info.pgvector_version >= (0, 5, 0)
    async with pool.connection() as conn:
        tenants = await (await conn.execute("SELECT name FROM tenants")).fetchall()
        dbs = await (await conn.execute("SELECT id::text, tenant, name FROM databases")).fetchall()
    assert [t[0] for t in tenants] == ["default_tenant"]
    assert dbs == [(ZERO_UUID, "default_tenant", "default_database")]
