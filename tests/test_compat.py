"""Differential tests: the same chromadb client calls against real Chroma and pgchroma.

Every scenario must produce the same normalized result (or the same exception class) on both
servers. Collection ids and timestamps differ by nature and are left out.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from harness import assert_same, outcome
from test_filters import DOC_CASES, ROWS, WHERE_CASES

pytest.importorskip("chromadb")

ALL = ["embeddings", "documents", "metadatas", "uris"]
QALL = ["embeddings", "documents", "metadatas", "uris", "distances"]


def info(c) -> dict:
    model = c._model
    return {
        "name": model.name,
        "metadata": model.metadata,
        "configuration": model.configuration_json,
        "dimension": model.dimension,
        "tenant": model.tenant,
        "database": model.database,
    }


def same(clients, fn):
    real, ours = clients
    expected, got = outcome(fn, real), outcome(fn, ours)
    assert_same(got, expected)
    return got


def new(client, name="docs", **kw):
    kw.setdefault("embedding_function", None)
    return client.create_collection(name, **kw)


def seed(client, name="docs", space=None):
    kw = {"configuration": {"hnsw": {"space": space}}} if space else {}
    c = new(client, name, **kw)
    c.add(
        ids=["a", "b", "c", "d"],
        embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
        documents=["apple pie", "banana split", "cherry tart", None],
        metadatas=[{"k": 1, "s": "x"}, {"k": 2, "s": "y"}, {"k": 3}, None],
        uris=["u-a", None, "u-c", None],
    )
    return c


def seed_filters(client):
    c = new(client, "filters")
    c.add(
        ids=[r[0] for r in ROWS],
        embeddings=[[float(i), 1.0] for i in range(len(ROWS))],
        metadatas=[r[1] for r in ROWS],
        documents=[r[2] for r in ROWS],
    )
    return c


# ----------------------------------------------------------------- collections


def test_create_and_get_collection(clients):
    same(
        clients,
        lambda cl: (
            info(new(cl, metadata={"topic": "t", "n": 1})),
            info(cl.get_collection("docs")),
        ),
    )


def test_default_embedding_function_config_roundtrip(clients):
    same(clients, lambda cl: (info(cl.create_collection("cfg")), info(cl.get_collection("cfg"))))


@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_configuration_spaces(clients, space):
    same(
        clients, lambda cl: info(new(cl, configuration={"hnsw": {"space": space, "ef_search": 50}}))
    )


def test_legacy_metadata_space(clients):
    def run(cl):
        c = new(cl, metadata={"hnsw:space": "cosine"})
        c.add(ids=["a", "b"], embeddings=[[1, 0], [1, 1]])
        return info(c), c.query(query_embeddings=[[1, 0]], n_results=2, include=["distances"])

    same(clients, run)


def test_duplicate_missing_and_invalid(clients):
    def run(cl):
        new(cl)
        out = []
        for fn in (
            lambda: new(cl),
            lambda: cl.get_collection("nope"),
            lambda: cl.delete_collection("nope"),
            lambda: new(cl, "x"),
            lambda: new(cl, "-bad-"),
        ):
            try:
                fn()
                out.append("ok")
            except Exception as e:  # noqa: BLE001
                out.append(type(e).__name__)
        return out

    same(clients, run)


def test_get_or_create_keeps_existing(clients):
    def run(cl):
        new(cl, metadata={"v": 1})
        return info(cl.get_or_create_collection("docs", metadata={"v": 2}, embedding_function=None))

    same(clients, run)


def test_modify_list_count_delete(clients):
    def run(cl):
        for n in ("aaa", "bbb", "ccc"):
            new(cl, n)
        cl.get_collection("bbb").modify(name="bbb2", metadata={"m": True})
        listed = cl.list_collections()
        # Chroma lists collections ordered by id; ids are random, so compare the rule.
        ordered_by_id = [str(c.id) for c in listed] == sorted(str(c.id) for c in listed)
        count = cl.count_collections()
        cl.delete_collection("aaa")
        names = sorted(c.name for c in listed)
        return names, ordered_by_id, count, cl.count_collections(), info(cl.get_collection("bbb2"))

    same(clients, run)


def test_list_collections_paging(clients):
    def run(cl):
        for n in ("c01", "c02", "c03", "c04"):
            new(cl, n)
        full = [c.name for c in cl.list_collections()]
        page = [c.name for c in cl.list_collections(limit=2, offset=1)]
        return page == full[1:3], len(page)

    same(clients, run)


# --------------------------------------------------------------------- records


def test_add_and_get_everything(clients):
    same(clients, lambda cl: seed(cl).get(include=ALL))


def test_add_existing_id_is_ignored(clients):
    def run(cl):
        c = seed(cl)
        c.add(ids=["a", "z"], embeddings=[[9, 9, 9], [5, 5, 5]], documents=["dup", "zed"])
        return c.get(include=ALL)

    same(clients, run)


def test_wrong_dimension_errors(clients):
    def run(cl):
        c = seed(cl)
        out = []
        for fn in (
            lambda: c.add(ids=["x"], embeddings=[[1, 2]]),
            lambda: c.query(query_embeddings=[[1, 2]], n_results=1),
            lambda: c.upsert(ids=["a"], embeddings=[[1, 2, 3, 4]]),
        ):
            try:
                fn()
                out.append("ok")
            except Exception as e:  # noqa: BLE001
                out.append(type(e).__name__)
        return out

    same(clients, run)


def test_get_ids_limit_offset(clients):
    def run(cl):
        c = seed(cl)
        return (
            c.get(ids=["d", "a", "missing"]),
            c.get(limit=2),
            c.get(offset=2),
            c.get(limit=1, offset=1),
            c.get(limit=10, offset=10),
            c.peek(2),
        )

    same(clients, run)


@pytest.mark.parametrize("where,_", WHERE_CASES, ids=[str(c[0]) for c in WHERE_CASES])
def test_where_live(clients, where, _):
    same(clients, lambda cl: seed_filters(cl).get(where=where, include=[])["ids"])


@pytest.mark.parametrize("where_document,_", DOC_CASES, ids=[str(c[0]) for c in DOC_CASES])
def test_where_document_live(clients, where_document, _):
    same(clients, lambda cl: seed_filters(cl).get(where_document=where_document, include=[])["ids"])


def test_where_number_types(clients):
    def run(cl):
        c = new(cl)
        c.add(
            ids=["i", "f", "b", "s", "big", "neg"],
            embeddings=[[1, 0]] * 6,
            metadatas=[{"v": 2}, {"v": 2.0}, {"v": False}, {"v": "2"}, {"v": 2**40}, {"v": -3}],
        )
        return [
            c.get(where=w, include=[])["ids"]
            for w in (
                {"v": 2},
                {"v": 2.0},
                {"v": False},
                {"v": 0},
                {"v": {"$gt": 2}},
                {"v": {"$lt": 2.5}},
                {"v": {"$gte": -3.5}},
                {"v": {"$in": [2, 2**40]}},
                {"v": {"$nin": ["2"]}},
            )
        ]

    same(clients, run)


def test_document_contains_special_chars(clients):
    docs = [
        "50% off",
        "snake_case",
        "C:\\path\\x",
        "it's",
        'say "hi"',
        "naïve café",
        "emoji 🎉 ok",
        "[a]{b}(c)*+?",
    ]

    def run(cl):
        c = new(cl)
        c.add(
            ids=[str(i) for i in range(len(docs))], embeddings=[[1, 0]] * len(docs), documents=docs
        )
        needles = ["%", "_", "\\", "'", '"', "é", "🎉", "{b}", "(c)*", "a]"]
        return [c.get(where_document={"$contains": n}, include=[])["ids"] for n in needles]

    same(clients, run)


@pytest.mark.parametrize("space", ["l2", "cosine", "ip"])
def test_query_per_space(clients, space):
    same(
        clients,
        lambda cl: seed(cl, space=space).query(
            query_embeddings=[[1, 0.1, 0], [0.1, 0.6, 1]], n_results=3, include=QALL
        ),
    )


def test_query_empty_and_overflow(clients):
    def run(cl):
        empty = new(cl, "empty")
        e = empty.query(query_embeddings=[[1, 2]], n_results=3, include=["distances", "documents"])
        full = seed(cl)
        return e, full.query(query_embeddings=[[1, 0, 0]], n_results=50)

    same(clients, run)


def test_query_with_filters_and_ids(clients):
    def run(cl):
        c = seed(cl)
        return (
            c.query(query_embeddings=[[1, 0, 0]], n_results=2, where={"k": {"$gte": 2}}),
            c.query(query_embeddings=[[1, 0, 0]], n_results=5, where_document={"$contains": "an"}),
            c.query(
                query_embeddings=[[0, 1, 0]], n_results=5, ids=["a", "c"], include=["distances"]
            ),
        )

    same(clients, run)


def test_update_semantics(clients):
    def run(cl):
        c = seed(cl)
        c.update(
            ids=["a", "b", "missing"], metadatas=[{"k": 10, "s": None}, {"new": "v"}, {"q": 1}]
        )
        c.update(ids=["c"], embeddings=[[0, 0, 2]], documents=["cherry 2"], uris=["u-c2"])
        return c.get(include=ALL)

    same(clients, run)


def test_upsert_semantics(clients):
    def run(cl):
        c = seed(cl)
        c.upsert(
            ids=["b", "e"],
            embeddings=[[0, 2, 0], [2, 0, 2]],
            metadatas=[{"k": None, "z": 1}, {"z": 2, "gone": None}],
            documents=["banana 2", "elder"],
        )
        c.upsert(ids=["c"], embeddings=[[0, 0, 3]])
        return c.get(include=ALL)

    same(clients, run)


def test_delete_semantics(clients):
    def run(cl):
        c = seed(cl)
        out = [
            c.delete(ids=["a", "b", "c"], where_document={"$contains": "an"}),
            c.delete(where={"k": 3}),
            c.count(),
        ]
        return out, c.get(include=["documents"])

    same(clients, run)


def test_metadata_value_roundtrip(clients):
    metas = [
        {"int": 7, "neg": -42, "big": 2**53, "float": 0.1, "whole_float": 3.0, "tiny": 1e-7},
        {"bool_t": True, "bool_f": False, "empty_str": "", "unicode_ключ": "значение 日本"},
        {"ints": [1, 2, 3], "strs": ["a", "b"], "floats": [0.5, 1.5], "bools": [True, False]},
    ]

    def run(cl):
        c = new(cl, "meta-types")
        c.add(ids=["0", "1", "2"], embeddings=[[1, 0]] * 3, metadatas=metas)
        r = c.get(include=["metadatas"])
        return r, [{k: type(v).__name__ for k, v in m.items()} for m in r["metadatas"]]

    same(clients, run)


def test_documents_and_uris_roundtrip(clients):
    docs = ["", " leading space", "multi\nline\ttext", "x" * 10_000, "nul-free \u2028 sep"]

    def run(cl):
        c = new(cl, "docs-uris")
        c.add(
            ids=[str(i) for i in range(5)],
            embeddings=[[1.5, -2.25]] * 5,
            documents=docs,
            uris=[f"s3://b/{i}" for i in range(5)],
        )
        return c.get(include=ALL)

    same(clients, run)


def test_large_batch_base64(clients):
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(1200, 64)).astype(np.float32)

    def run(cl):
        c = new(cl)
        c.add(
            ids=[f"r{i}" for i in range(1200)],
            embeddings=vecs,
            metadatas=[{"i": i, "even": i % 2 == 0} for i in range(1200)],
        )
        return (
            c.count(),
            c.query(query_embeddings=vecs[:3], n_results=5, include=["distances"]),
            c.query(
                query_embeddings=vecs[10:11],
                n_results=5,
                where={"even": True},
                include=["distances"],
            ),
        )

    same(clients, run)


def test_tenants_and_databases(clients):
    import chromadb
    from chromadb.config import Settings

    tenant = f"t-{uuid.uuid4().hex[:8]}"

    def run(cl):
        settings = cl.get_settings()
        admin = chromadb.AdminClient(
            Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=settings.chroma_server_host,
                chroma_server_http_port=settings.chroma_server_http_port,
                anonymized_telemetry=False,
            )
        )
        admin.create_tenant(tenant)
        admin.create_database("db-one", tenant=tenant)
        admin.create_database("db-two", tenant=tenant)
        dbs = sorted(d["name"] for d in admin.list_databases(tenant=tenant))
        admin.delete_database("db-one", tenant=tenant)
        remaining = [d["name"] for d in admin.list_databases(tenant=tenant)]
        errors = []
        for fn in (lambda: admin.get_tenant("no-such-tenant"), lambda: admin.create_tenant(tenant)):
            try:
                fn()
                errors.append("ok")
            except Exception as e:  # noqa: BLE001
                errors.append(type(e).__name__)
        return admin.get_tenant(tenant)["name"], dbs, remaining, errors

    same(clients, run)


def test_collections_in_other_tenant_database(clients):
    import chromadb
    from chromadb.config import Settings

    tenant = f"t-{uuid.uuid4().hex[:8]}"

    def run(cl):
        s = cl.get_settings()
        admin = chromadb.AdminClient(
            Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=s.chroma_server_host,
                chroma_server_http_port=s.chroma_server_http_port,
                anonymized_telemetry=False,
            )
        )
        admin.create_tenant(tenant)
        admin.create_database("work", tenant=tenant)
        other = chromadb.HttpClient(
            host=s.chroma_server_host,
            port=s.chroma_server_http_port,
            tenant=tenant,
            database="work",
            settings=Settings(anonymized_telemetry=False),
        )
        c = new(other, "scoped")
        c.add(ids=["1"], embeddings=[[1, 2]])
        return (
            info(c),
            [x.name for x in cl.list_collections()],
            other.get_collection("scoped").count(),
        )

    same(clients, run)
