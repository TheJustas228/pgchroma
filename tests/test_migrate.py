import numpy as np
import pytest

from pgchroma.migrate import migrate, verify
from pgchroma.source import ChromaSource
from pgchroma.store import Store

pytest.importorskip("chromadb")

T, D = "default_tenant", "default_database"


def seed_source(url):
    from harness import make_client

    cl = make_client(url)
    rng = np.random.default_rng(11)
    specs = {"articles": ("cosine", 700), "products": ("l2", 260), "tiny-ip": ("ip", 3)}
    for name, (space, n) in specs.items():
        c = cl.create_collection(
            name,
            metadata={"owner": "team-a", "v": 2},
            configuration={"hnsw": {"space": space}},
            embedding_function=None,
        )
        c.add(
            ids=[f"{name}-{i}" for i in range(n)],
            embeddings=rng.normal(size=(n, 24)).astype(np.float32),
            documents=[f"{name} document {i}" if i % 7 else None for i in range(n)],
            metadatas=[
                {
                    "i": i,
                    "f": i / 3,
                    "even": i % 2 == 0,
                    "tags": ["x", "y"] if i % 5 == 0 else ["z"],
                }
                if i % 11
                else None
                for i in range(n)
            ],
            uris=[f"s3://bucket/{name}/{i}" if i % 3 == 0 else None for i in range(n)],
        )
    return specs


@pytest.fixture
async def store(pool):
    s = Store(pool, pool.pgchroma_info)
    yield s
    await s.wait_for_indexing()


async def test_migrate_and_verify(real_chroma, store):
    specs = seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        report = await migrate(src, store, batch_size=100)
        assert {c.name: c.copied for c in report.collections} == {k: v[1] for k, v in specs.items()}
        check = await verify(src, store, sample=40, k=5)
    assert check.ok, check.problems
    assert all(c.query_overlap >= 0.95 for c in check.collections)

    listed = {c["name"]: c for c in await store.list_collections(T, D, None, None)}
    assert listed["articles"]["metadata"] == {"owner": "team-a", "v": 2}
    assert listed["articles"]["configuration_json"]["hnsw"]["space"] == "cosine"
    art = await store.collection(T, D, listed["articles"]["id"])
    r = await store.get(
        art,
        ["articles-0", "articles-11", "articles-3"],
        None,
        None,
        None,
        None,
        ["metadatas", "documents", "uris"],
    )
    assert r["ids"] == ["articles-0", "articles-3", "articles-11"]
    # articles-0 has a uri but no metadata ({}); articles-11 has neither (None)
    assert r["metadatas"] == [{}, {"i": 3, "f": 1.0, "even": False, "tags": ["z"]}, None]
    assert r["documents"] == [None, "articles document 3", "articles document 11"]


class FlakySource(ChromaSource):
    """Fails once, after a few pages, like a dropped connection."""

    def __init__(self, url, fail_after):
        super().__init__(url)
        self.pages = 0
        self.fail_after = fail_after

    async def get_page(self, *args, **kwargs):
        self.pages += 1
        if self.pages == self.fail_after:
            raise ConnectionError("simulated network failure")
        return await super().get_page(*args, **kwargs)


async def test_interrupted_migration_resumes_without_duplicates(real_chroma, store):
    seed_source(real_chroma.url)
    async with FlakySource(real_chroma.url, fail_after=5) as src:
        with pytest.raises(ConnectionError):
            await migrate(src, store, batch_size=100, concurrency=1)
    async with ChromaSource(real_chroma.url) as src:
        report = await migrate(src, store, batch_size=100)
        assert sum(c.copied for c in report.collections) == 963
        assert any(c.resumed for c in report.collections)
        check = await verify(src, store, sample=20, k=5)
    assert check.ok, check.problems


async def test_migrate_into_other_database_and_filters(real_chroma, store):
    seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        report = await migrate(
            src, store, collections=["products"], to_database="imported", batch_size=500
        )
    assert [c.name for c in report.collections] == ["products"]
    assert [c["name"] for c in await store.list_collections(T, "imported", None, None)] == [
        "products"
    ]


async def test_existing_target_collection_is_not_overwritten(real_chroma, store):
    seed_source(real_chroma.url)
    await store.create_collection(T, D, "tiny-ip", None, None, None, False)
    c = await store.collection(T, D, (await store.get_collection(T, D, "tiny-ip"))["id"])
    await store.add(c, ["mine"], np.ones((1, 24), dtype=np.float32), None, None, None)
    async with ChromaSource(real_chroma.url) as src:
        with pytest.raises(Exception, match="already exists"):
            await migrate(src, store, collections=["tiny-ip"])
