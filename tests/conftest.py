import os
import tempfile
import uuid

import pytest

from pgchroma.db import bootstrap, open_pool


@pytest.fixture(scope="session")
def database_url():
    """A Postgres with pgvector: $DATABASE_URL if set, else an embedded pgserver."""
    url = os.environ.get("DATABASE_URL")
    if url:
        yield url
        return
    import pgserver

    datadir = tempfile.mkdtemp(prefix="pgchroma-test-")
    server = pgserver.get_server(datadir, cleanup_mode="delete")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture
async def pool(database_url):
    """A connection pool bound to a fresh, bootstrapped schema that is dropped afterwards."""
    schema = f"t_{uuid.uuid4().hex[:12]}"
    p = await open_pool(database_url, schema)
    await bootstrap(p, schema)
    yield p
    async with p.connection() as conn:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await p.close()
