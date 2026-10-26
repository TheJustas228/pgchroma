"""Rough side-by-side numbers: real Chroma 1.5.9 vs pgchroma, same client, same data.

    uv run python bench/bench.py --n 50000 --dim 384

Uses an embedded Postgres (pgserver) unless DATABASE_URL is set. Numbers depend heavily on
hardware and Postgres settings; treat them as a sanity check, not a leaderboard.
"""

import argparse
import os
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from harness import ChromaProcess, PgChromaThread, make_client  # noqa: E402


def timed(fn, repeat):
    out = []
    for i in range(repeat):
        t = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out), sorted(out)[int(len(out) * 0.95) - 1]


def run(name, url, vecs, queries, batch):
    cl = make_client(url)
    for c in cl.list_collections():
        cl.delete_collection(c.name)
    col = cl.create_collection("bench", embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
    n = len(vecs)
    t = time.perf_counter()
    for s in range(0, n, batch):
        col.add(
            ids=[str(i) for i in range(s, min(n, s + batch))],
            embeddings=vecs[s : s + batch],
            metadatas=[{"bucket": i % 100, "even": i % 2 == 0} for i in range(s, min(n, s + batch))],
        )
    ingest = time.perf_counter() - t
    # The first search starts pgchroma's background HNSW build; wait until both servers
    # report indexing complete so queries are compared on equal terms.
    t = time.perf_counter()
    col.query(query_embeddings=queries[:1], n_results=10)
    while True:
        try:
            if col.get_indexing_status().op_indexing_progress >= 1.0:
                break
        except Exception:  # noqa: BLE001 - servers without the endpoint
            time.sleep(2)
            break
        time.sleep(0.5)
    index_wait = time.perf_counter() - t
    q50, q95 = timed(lambda i: col.query(query_embeddings=queries[i : i + 1], n_results=10), len(queries))
    f50, f95 = timed(lambda i: col.query(query_embeddings=queries[i : i + 1], n_results=10, where={"bucket": 7}), len(queries))
    g50, g95 = timed(lambda i: col.get(where={"bucket": i % 100}, limit=20), len(queries))
    return name, n / ingest, index_wait, q50, q95, f50, f95, g50, g95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)

    db = os.environ.get("DATABASE_URL")
    if not db:
        import pgserver

        db = pgserver.get_server(tempfile.mkdtemp(), cleanup_mode="delete").get_uri()
    real, ours = ChromaProcess(), PgChromaThread(db)
    try:
        rows = [run("chroma 1.5.9", real.url, vecs, queries, args.batch), run("pgchroma", ours.url, vecs, queries, args.batch)]
    finally:
        real.stop()
        ours.stop()
    print(f"\n{args.n:,} x {args.dim}-dim vectors, cosine, batch {args.batch}, {args.queries} queries\n")
    print(f"{'server':<14}{'ingest rec/s':>14}{'index wait':>12}{'query p50':>11}{'p95':>8}{'filtered p50':>14}{'p95':>8}{'get p50':>9}{'p95':>8}")
    for name, ips, wait, q50, q95, f50, f95, g50, g95 in rows:
        print(f"{name:<14}{ips:>14,.0f}{wait:>11.1f}s{q50:>9.1f}ms{q95:>6.1f}ms{f50:>12.1f}ms{f95:>6.1f}ms{g50:>7.1f}ms{g95:>6.1f}ms")


if __name__ == "__main__":
    main()
