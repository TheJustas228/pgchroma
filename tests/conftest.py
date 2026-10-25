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


# ---------------------------------------------------------------- real servers


@pytest.fixture
def real_chroma():
    """A fresh reference Chroma server per test (~1s), so no state carries between tests."""
    pytest.importorskip("chromadb")
    from harness import ChromaProcess

    try:
        proc = ChromaProcess()
    except FileNotFoundError:
        pytest.skip("the real Chroma server binary is not available")
    yield proc
    proc.stop()


@pytest.fixture(scope="session")
def pgchroma_server(database_url):
    from harness import PgChromaThread

    server = PgChromaThread(database_url, allow_reset=True)
    yield server
    server.stop()


def _clean(client):
    for c in client.list_collections():
        client.delete_collection(c.name)


@pytest.fixture
def clients(real_chroma, pgchroma_server):
    """(real Chroma client, pgchroma client), both starting with no collections."""
    from harness import make_client

    real, ours = make_client(real_chroma.url), make_client(pgchroma_server.url)
    _clean(ours)
    return real, ours
