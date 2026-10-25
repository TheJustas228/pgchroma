"""Chroma's Search API (`collection.search(Search()...)`) through the official client.

Single-node Chroma does not implement /search, so these tests check documented semantics:
Knn scores are distances (lower is better) or 0-based ranks, Rrf = -sum(w / (k + rank)),
records outside a Knn's top `limit` get its `default` or are dropped when it is None.
"""

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")
from chromadb import K, Knn, Rrf, Search  # noqa: E402

VECS = np.array(
    [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.3], [0, 0, 1], [0.5, 0.5, 0.5]],
    dtype=np.float32,
)
IDS = ["a", "b", "c", "d", "e", "f"]
META = [
    {"cat": "x", "year": 2019, "title": "A"},
    {"cat": "x", "year": 2021, "title": "B"},
    {"cat": "y", "year": 2022, "title": "C"},
    {"cat": "y", "year": 2018, "title": "D"},
    {"cat": "x", "year": 2023, "title": "E"},
    {"cat": "z", "year": 2020, "title": "F"},
]
DOCS = ["alpha one", "beta two", "gamma three", "delta four", "epsilon five", "zeta six"]


@pytest.fixture
def coll(pgchroma_server):
    from harness import make_client

    cl = make_client(pgchroma_server.url)
    for c in cl.list_collections():
        cl.delete_collection(c.name)
    c = cl.create_collection("search", embedding_function=None)
    c.add(ids=IDS, embeddings=VECS, metadatas=META, documents=DOCS)
    return c


def l2(q):
    return ((VECS - np.asarray(q, dtype=np.float32)) ** 2).sum(1)


def test_knn_with_filter_select_and_paging(coll):
    q = [1, 0.2, 0]
    s = (
        Search()
        .where((K("cat") == "x") & (K("year") >= 2020))
        .rank(Knn(query=q))
        .limit(5)
        .select(K.DOCUMENT, K.SCORE, "title")
    )
    r = coll.search(s)
    d = l2(q)
    expected = sorted([i for i in (1, 4)], key=lambda i: d[i])
    assert r["ids"] == [[IDS[i] for i in expected]]
    assert np.allclose(r["scores"][0], [d[i] for i in expected], atol=1e-5)
    assert r["documents"][0] == [DOCS[i] for i in expected]
    assert r["metadatas"][0] == [{"title": META[i]["title"]} for i in expected]

    r = coll.search(Search().rank(Knn(query=q)).limit(2, offset=1).select(K.SCORE))
    order = list(np.argsort(d))
    assert r["ids"] == [[IDS[i] for i in order[1:3]]]


def test_knn_limit_bounds_candidates(coll):
    r = coll.search(Search().rank(Knn(query=[1, 0, 0], limit=3)).limit(10))
    assert len(r["ids"][0]) == 3


def test_filter_only_returns_insertion_order(coll):
    r = coll.search(Search().where(K("cat") == "y").select(K.METADATA))
    assert r["ids"] == [["c", "d"]]
    assert r["metadatas"][0] == [META[2], META[3]]


def test_special_keys_in_filter(coll):
    r = coll.search(Search().where(K.ID.is_in(["a", "c", "e"]) & K.DOCUMENT.contains("a")))
    assert r["ids"] == [["a", "c"]]
    r = coll.search(Search().where(K.DOCUMENT.not_contains("a")))
    assert r["ids"] == [["e"]]


def test_arithmetic_on_scores(coll):
    q = [0, 1, 0]
    r = coll.search(Search().rank(Knn(query=q) * 0.5 + 1).limit(3).select(K.SCORE))
    d = np.sort(l2(q))[:3]
    assert np.allclose(r["scores"][0], 0.5 * d + 1, atol=1e-5)


def test_rrf_fuses_ranks(coll):
    q1, q2 = [1, 0, 0], [0, 0, 1]
    s = (
        Search()
        .rank(
            Rrf(
                [
                    Knn(query=q1, return_rank=True, limit=6),
                    Knn(query=q2, return_rank=True, limit=6),
                ],
                weights=[2.0, 1.0],
                k=60,
            )
        )
        .limit(6)
        .select(K.SCORE)
    )
    r = coll.search(s)
    rank1 = {i: r_ for r_, i in enumerate(np.argsort(l2(q1), kind="stable"))}
    rank2 = {i: r_ for r_, i in enumerate(np.argsort(l2(q2), kind="stable"))}
    scores = {i: -(2.0 / (60 + rank1[i]) + 1.0 / (60 + rank2[i])) for i in range(6)}
    expected = sorted(scores, key=lambda i: scores[i])
    assert r["ids"][0] == [IDS[i] for i in expected]
    assert np.allclose(r["scores"][0], [scores[i] for i in expected], atol=1e-6)


def test_default_keeps_records_outside_knn(coll):
    r = coll.search(Search().rank(Knn(query=[1, 0, 0], limit=2)).limit(10))
    assert len(r["ids"][0]) == 2  # no default: records outside the top 2 are dropped
    # with a default, a combination with another Knn can keep them
    s = (
        Search()
        .rank(Knn(query=[1, 0, 0], limit=2, default=10.0) + Knn(query=[0, 0, 1], limit=6))
        .limit(10)
        .select(K.SCORE)
    )
    r = coll.search(s)
    assert len(r["ids"][0]) == 6


def test_batch_of_searches(coll):
    r = coll.search([Search().where(K("cat") == "z"), Search().rank(Knn(query=[0, 0, 1])).limit(1)])
    assert r["ids"] == [["f"], ["e"]]


def test_sparse_key_is_reported_as_unsupported(coll):
    with pytest.raises(Exception, match="not supported"):
        coll.search(Search().rank(Knn(query=[1, 0, 0], key="sparse_field")))
