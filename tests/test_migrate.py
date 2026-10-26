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


class DegradedSource(ChromaSource):
    """Returns only the first `keep` of the true neighbours, like a low-recall ANN index."""

    def __init__(self, url, keep):
        super().__init__(url)
        self.keep = keep

    async def query(self, database, collection_id, embeddings, n_results):
        r = await super().query(database, collection_id, embeddings, n_results)
        r["ids"] = [
            ids[: self.keep] + [f"other-{i}" for i in range(n_results - self.keep)]
            for ids in r["ids"]
        ]
        return r


@pytest.mark.parametrize("keep,ok,warned", [(10, True, False), (7, True, True), (3, False, False)])
async def test_verify_grades_query_overlap(real_chroma, store, keep, ok, warned):
    seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        await migrate(src, store, collections=["articles"])
    async with DegradedSource(real_chroma.url, keep) as src:
        check = await verify(src, store, collections=["articles"], sample=20, k=10)
    assert check.ok is ok, check.problems
    assert bool(check.warnings) is warned


class TiedSource(ChromaSource):
    """Answers queries with other copies of the same embedding: equally near, different ids."""

    copies = 30

    async def query(self, database, collection_id, embeddings, n_results):
        r = await super().query(database, collection_id, embeddings, n_results)
        groups = [ids[0].split("-")[1] for ids in r["ids"]]
        r["ids"] = [[f"dup-{g}-{self.copies - 1 - j}" for j in range(n_results)] for g in groups]
        return r


async def test_verify_counts_equally_near_duplicates_as_exact(real_chroma, store):
    from harness import make_client

    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(4, 16)).astype(np.float32)
    c = make_client(real_chroma.url).create_collection(
        "dups", configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    c.add(
        ids=[f"dup-{g}-{j}" for g in range(4) for j in range(TiedSource.copies)],
        embeddings=np.repeat(vectors, TiedSource.copies, axis=0),
    )
    async with ChromaSource(real_chroma.url) as src:
        await migrate(src, store, collections=["dups"])
    async with TiedSource(real_chroma.url) as src:
        check = await verify(src, store, collections=["dups"], sample=20, k=10)
    assert check.ok, check.problems
    assert not check.warnings
    assert check.collections[0].query_overlap == 1.0


async def test_migrate_builds_indexes_for_large_collections(real_chroma, pool):
    seed_source(real_chroma.url)
    store = Store(pool, pool.pgchroma_info, index_threshold=200)
    async with ChromaSource(real_chroma.url) as src:
        report = await migrate(src, store)
    indexed = {c.name: c.indexed for c in report.collections}
    assert indexed == {"articles": True, "products": True, "tiny-ip": False}
    for name in ("articles", "products"):
        ref = await store.collection(T, D, (await store.get_collection(T, D, name))["id"])
        assert await store.indexing_progress(ref) == 1.0


QWEN = {
    "type": "known",
    "name": "chroma-cloud-qwen",
    "config": {"model": "Qwen/Qwen3-Embedding-0.6B", "task": None},
}
SPLADE = {
    "type": "known",
    "name": "chroma-cloud-splade",
    "config": {"model": "prithivida/Splade_PP_en_v1", "include_tokens": False},
}
SPLADE_SCHEMA = {
    "defaults": {},
    "keys": {
        "splade": {
            "sparse_vector": {
                "sparse_vector_index": {
                    "enabled": True,
                    "config": {"embedding_function": SPLADE, "source_key": "#document"},
                }
            }
        }
    },
}


def test_localize_embedding_functions():
    import copy

    from pgchroma.migrate import localize_embedding_functions

    cfg = {"hnsw": {"space": "cosine"}, "embedding_function": QWEN}
    schema = copy.deepcopy(SPLADE_SCHEMA)
    schema["keys"]["#embedding"] = {
        "float_list": {
            "vector_index": {
                "enabled": True,
                "config": {
                    "space": "cosine",
                    "embedding_function": QWEN,
                    "source_key": "#document",
                },
            }
        }
    }
    before = copy.deepcopy(schema)
    new_cfg, new_schema, notes = localize_embedding_functions(cfg, schema)
    assert new_cfg["embedding_function"] == {
        "type": "known",
        "name": "sentence_transformer",
        "config": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "device": "cpu",
            "normalize_embeddings": True,
            "kwargs": {"model_kwargs": {"torch_dtype": "float32"}},
        },
    }
    assert new_cfg["hnsw"] == {"space": "cosine"}
    vector = new_schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]
    assert vector["embedding_function"] == new_cfg["embedding_function"]
    assert vector["space"] == "cosine"
    sparse = new_schema["keys"]["splade"]["sparse_vector"]["sparse_vector_index"]["config"]
    assert sparse == {
        "embedding_function": {
            "type": "known",
            "name": "huggingface_sparse",
            "config": {
                "model_name": "prithivida/Splade_PP_en_v1",
                "device": "cpu",
                "task": "document",
                "query_config": {"task": "document"},
                "kwargs": {},
            },
        },
        "source_key": "#document",
    }
    assert schema == before  # inputs are not modified
    assert len(notes) == 2
    assert "Qwen/Qwen3-Embedding-0.6B" in notes[0] and "sentence-transformers" in notes[0]
    assert "prithivida/Splade_PP_en_v1" in notes[1]

    task = {"embedding_function": {**QWEN, "config": {**QWEN["config"], "task": "nl_to_code"}}}
    _, _, notes = localize_embedding_functions(task, None)
    assert "nl_to_code" in notes[0] and "examples/local_qwen_embeddings.py" in notes[0]

    plain = {"embedding_function": None}
    assert localize_embedding_functions(plain, None) == (plain, None, [])


class CloudSchemaSource(ChromaSource):
    """Lists collections the way Chroma Cloud does for one with a hosted SPLADE index
    (single-node Chroma refuses sparse indexes)."""

    async def collections(self, database):
        import copy

        out = await super().collections(database)
        for c in out:
            c["schema"] = copy.deepcopy(SPLADE_SCHEMA)
        return out


async def test_migrate_localizes_hosted_sparse_embedding_function(real_chroma, store):
    from harness import make_client

    c = make_client(real_chroma.url).create_collection("hybrid", embedding_function=None)
    c.add(ids=["x"], embeddings=[[1.0, 0.0]], documents=["hello"])
    async with CloudSchemaSource(real_chroma.url) as src:
        report = await migrate(src, store, collections=["hybrid"])
    assert any("chroma-cloud-splade" in w for w in report.collections[0].warnings)
    async with CloudSchemaSource(real_chroma.url) as src:
        report = await migrate(
            src, store, collections=["hybrid"], to_database="local", local_embedding_functions=True
        )
    assert any("prithivida/Splade_PP_en_v1" in w for w in report.collections[0].warnings)
    got = await store.get_collection(T, "local", "hybrid")
    index = got["schema"]["keys"]["splade"]["sparse_vector"]["sparse_vector_index"]
    assert index["config"]["embedding_function"]["name"] == "huggingface_sparse"
    assert index["config"]["source_key"] == "#document"


async def test_migrate_rewrites_hosted_embedding_function_on_request(real_chroma, store):
    import httpx

    ef = {
        "type": "known",
        "name": "chroma-cloud-qwen",
        "config": {"model": "Qwen/Qwen3-Embedding-0.6B", "task": None},
    }
    base = f"{real_chroma.url}/api/v2/tenants/default_tenant/databases/default_database/collections"
    r = httpx.post(
        base,
        json={
            "name": "cloudy",
            "configuration": {"hnsw": {"space": "cosine"}, "embedding_function": ef},
        },
    )
    assert r.status_code == 200, r.text
    httpx.post(f"{base}/{r.json()['id']}/add", json={"ids": ["x"], "embeddings": [[1.0, 0.0]]})
    async with ChromaSource(real_chroma.url) as src:
        report = await migrate(src, store, collections=["cloudy"], local_embedding_functions=True)
    got = await store.get_collection(T, D, "cloudy")
    assert got["configuration_json"]["embedding_function"]["name"] == "sentence_transformer"
    assert got["configuration_json"]["hnsw"]["space"] == "cosine"
    assert any("Qwen/Qwen3-Embedding-0.6B" in w for w in report.collections[0].warnings)
