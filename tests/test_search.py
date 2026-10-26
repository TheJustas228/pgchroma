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


# Sparse vectors live in metadata. Chroma scores them as 1 - query·document (lower is better);
# only records sharing a dimension with the query are candidates. With bm25 the query weights
# are scaled by ln(1 + (N - n_t + 0.5) / (n_t + 0.5)), N = records, n_t = records with dimension t.
# (rust/worker/src/execution/operators/{sparse_log_knn,idf}.rs)

SPARSE = {
    "a": ([1, 4, 9], [0.5, 1.0, 2.0]),
    "b": ([4], [3.0]),
    "c": ([2, 9], [1.0, 0.25]),
    "d": None,
    "e": ([7], [5.0]),
    "f": ([1, 2, 4], [1.0, 1.0, 0.1]),
}


def sparse_metadatas():
    from chromadb import SparseVector

    out = []
    for rid, meta in zip(IDS, META, strict=True):
        m = dict(meta)
        if SPARSE[rid]:
            m["sp"] = SparseVector(indices=SPARSE[rid][0], values=SPARSE[rid][1])
        out.append(m)
    return out


def sparse_scores(q_idx, q_val, idf=False):
    weights = dict(zip(q_idx, q_val, strict=True))
    if idf:
        n = len(IDS)
        for t in weights:
            nt = sum(1 for v in SPARSE.values() if v and t in v[0])
            weights[t] *= np.log1p((n - nt + 0.5) / (nt + 0.5))
    scores = {}
    for rid in IDS:
        if SPARSE[rid] and set(SPARSE[rid][0]) & set(weights):
            dot = sum(weights.get(i, 0.0) * v for i, v in zip(*SPARSE[rid], strict=True))
            scores[rid] = 1 - dot
    return scores


def ranked(scores):
    return sorted(scores, key=lambda rid: (scores[rid], IDS.index(rid)))


@pytest.fixture
def sparse_coll(pgchroma_server):
    from harness import make_client

    cl = make_client(pgchroma_server.url)
    for c in cl.list_collections():
        cl.delete_collection(c.name)
    c = cl.create_collection("sparse", embedding_function=None)
    c.add(ids=IDS, embeddings=VECS, metadatas=sparse_metadatas(), documents=DOCS)
    return c


def test_sparse_vectors_round_trip_in_metadata(sparse_coll):
    got = sparse_coll.get(ids=["a", "d"], include=["metadatas"])
    assert got["metadatas"][0]["sp"].indices == [1, 4, 9]
    assert got["metadatas"][0]["sp"].values == [0.5, 1.0, 2.0]
    assert "sp" not in got["metadatas"][1]


def test_sparse_knn_scores_are_one_minus_dot(sparse_coll):
    from chromadb import SparseVector

    q = ([1, 2, 4], [1.0, 0.5, 0.2])
    r = sparse_coll.search(
        Search().rank(Knn(query=SparseVector(*q), key="sp", limit=10)).select(K.SCORE)
    )
    want = sparse_scores(*q)
    assert r["ids"] == [ranked(want)]
    assert np.allclose(r["scores"][0], [want[i] for i in ranked(want)], atol=1e-6)


def test_sparse_knn_with_filter_and_limit(sparse_coll):
    from chromadb import SparseVector

    q = ([4, 9], [1.0, 1.0])
    r = sparse_coll.search(
        Search().where(K("cat") == "x").rank(Knn(query=SparseVector(*q), key="sp", limit=2))
    )
    want = {k: v for k, v in sparse_scores(*q).items() if META[IDS.index(k)]["cat"] == "x"}
    assert r["ids"] == [ranked(want)[:2]]


def test_bm25_scales_query_weights_by_idf(pgchroma_server):
    from chromadb import Schema, SparseVector, SparseVectorIndexConfig
    from harness import make_client

    cl = make_client(pgchroma_server.url)
    for c in cl.list_collections():
        cl.delete_collection(c.name)
    schema = Schema().create_index(config=SparseVectorIndexConfig(bm25=True), key="sp")
    c = cl.create_collection("bm25", schema=schema, embedding_function=None)
    c.add(ids=IDS, embeddings=VECS, metadatas=sparse_metadatas())
    q = ([1, 2, 4, 7], [1.0, 1.0, 1.0, 1.0])
    r = c.search(Search().rank(Knn(query=SparseVector(*q), key="sp", limit=10)).select(K.SCORE))
    want = sparse_scores(*q, idf=True)
    assert r["ids"] == [ranked(want)]
    assert np.allclose(r["scores"][0], [want[i] for i in ranked(want)], atol=1e-6)


def test_hybrid_rrf_of_dense_and_sparse(sparse_coll):
    from chromadb import SparseVector

    dq, sq = [1, 0, 0], ([4], [1.0])
    r = sparse_coll.search(
        Search()
        .rank(
            Rrf(
                [
                    Knn(query=dq, return_rank=True, limit=3),
                    Knn(query=SparseVector(*sq), key="sp", return_rank=True, limit=3),
                ],
                k=60,
            )
        )
        .select(K.SCORE)
    )
    dense = sorted(range(len(IDS)), key=lambda i: (l2(dq)[i], i))[:3]
    sparse = ranked(sparse_scores(*sq))[:3]
    both = [IDS[i] for i in dense if IDS[i] in sparse]
    fused = {
        rid: -(1 / (60 + [IDS[i] for i in dense].index(rid)) + 1 / (60 + sparse.index(rid)))
        for rid in both
    }
    assert r["ids"] == [ranked(fused)]
    assert np.allclose(r["scores"][0], [fused[i] for i in ranked(fused)], atol=1e-9)


def test_bm25_embedding_function_end_to_end(pgchroma_server):
    pytest.importorskip("snowballstemmer")
    from chromadb import Schema, SparseVectorIndexConfig
    from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
    from harness import make_client

    cl = make_client(pgchroma_server.url)
    for c in cl.list_collections():
        cl.delete_collection(c.name)
    schema = Schema().create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT, embedding_function=ChromaBm25EmbeddingFunction(), bm25=True
        ),
        key="bm25",
    )
    c = cl.create_collection("bm25-docs", schema=schema, embedding_function=None)
    docs = ["the cat sat on the mat", "dogs bark loudly", "a cat chased a cat", "fresh bread"]
    c.add(ids=["m", "d", "cc", "b"], embeddings=[[1, 0], [0, 1], [1, 1], [0, 0.5]], documents=docs)
    r = c.search(Search().rank(Knn(query="cats", key="bm25", limit=10)).select(K.DOCUMENT))
    assert r["ids"] == [["cc", "m"]]


def test_sparse_knn_rejects_dense_query_on_metadata_key(coll):
    with pytest.raises(chromadb.errors.InvalidArgumentError, match="sparse"):
        coll.search(Search().rank(Knn(query=[1, 0, 0], key="sparse_field")))


# group_by follows rust/worker/src/execution/operators/ranked_group_by.rs: after ranking, records
# are grouped by the group keys (records missing a key share the "missing" group), each group
# keeps its first k records ordered by the aggregate keys (MinK ascending, MaxK descending), and
# the survivors are re-sorted by score before limit/offset.


def grouped_reference(scores, group_key, agg_key, k, largest=False):
    groups = {}
    for rid in sorted(scores, key=lambda r: (scores[r], IDS.index(r))):
        groups.setdefault(META[IDS.index(rid)].get(group_key), []).append(rid)
    kept = []
    for members in groups.values():

        def order(r):
            return scores[r] if agg_key == "#score" else META[IDS.index(r)][agg_key]

        kept += sorted(members, key=order, reverse=largest)[:k]
    return sorted(kept, key=lambda r: (scores[r], IDS.index(r)))


def test_group_by_keeps_best_per_group(coll):
    from chromadb.execution.expression.operator import GroupBy, MinK

    q = [0.6, 0.5, 0.2]
    r = coll.search(
        Search()
        .rank(Knn(query=q, limit=6))
        .group_by(GroupBy(keys=K("cat"), aggregate=MinK(keys=K.SCORE, k=1)))
        .select(K.SCORE)
    )
    scores = dict(zip(IDS, l2(q).tolist(), strict=True))
    want = grouped_reference(scores, "cat", "#score", 1)
    assert len(want) == 3
    assert r["ids"] == [want]
    assert np.allclose(r["scores"][0], [scores[i] for i in want], atol=1e-5)


def test_group_by_max_k_on_metadata_then_limit(coll):
    from chromadb.execution.expression.operator import GroupBy, MaxK

    q = [0.2, 0.3, 0.9]
    r = coll.search(
        Search()
        .rank(Knn(query=q, limit=6))
        .group_by(GroupBy(keys=[K("cat")], aggregate=MaxK(keys=[K("year")], k=2)))
        .limit(3)
    )
    scores = dict(zip(IDS, l2(q).tolist(), strict=True))
    assert r["ids"] == [grouped_reference(scores, "cat", "year", 2, largest=True)[:3]]


def test_group_by_missing_key_is_one_group(coll):
    from chromadb.execution.expression.operator import GroupBy, MinK

    q = [0, 1, 0]
    r = coll.search(
        Search()
        .rank(Knn(query=q, limit=6))
        .group_by(GroupBy(keys=K("no_such_field"), aggregate=MinK(keys=K.SCORE, k=2)))
    )
    order = sorted(IDS, key=lambda rid: (l2(q)[IDS.index(rid)], IDS.index(rid)))
    assert r["ids"] == [order[:2]]
