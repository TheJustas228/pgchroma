# Changelog

## 0.4.0 (2026-10-27)

- Added: `pgchroma export` saves collections from Chroma Cloud, a self-hosted Chroma or a
  pgchroma server to JSON Lines files (one file per collection: a header line with the
  collection, then one line per record). Read-only on the source, resumable, no database
  needed.
- Added: `pgchroma migrate --from <export directory>` and `pgchroma verify --from <export
  directory>` load and check an export, with the same verification as a direct copy.
- Fixed: `pgchroma migrate` crashed with "Set of Tasks/Futures is empty" when the source had no
  collections.

## 0.3.0 (2026-10-27)

- Added: works behind transaction poolers (PgBouncer, Supavisor, Neon's pooled endpoint).
  pgchroma sets its search path per transaction there, builds indexes with standalone,
  schema-qualified statements and a lease row instead of a session advisory lock, and detects
  Neon `-pooler` hosts and port 6543 automatically (`PGCHROMA_POOLER=transaction` for others).
  Before, those connection strings failed with "Collection does not exist".
- Tested on Neon (Postgres 17.11, pgvector 0.8.0); CI runs the pooler tests through PgBouncer.
- Docs site: https://thejustas228.github.io/pgchroma/

## 0.2.0 (2026-10-26)

- Added: sparse vector search. `Knn` over `SparseVector` metadata fields with Chroma Cloud's
  scoring (`1 - query·document`), BM25 weighting when the schema sets `bm25=True`, and hybrid
  `Rrf` of dense and sparse results. Works with `ChromaBm25EmbeddingFunction` unchanged.
  Single-node Chroma rejects sparse vectors; pgchroma runs them on plain Postgres.
- Added: Search API `group_by` with `MinK` / `MaxK`, following Chroma's RankedGroupBy: top k
  records per group by the aggregate keys, then re-sorted by score before `limit`.
- Added: `--local-embedding-functions` also replaces Chroma Cloud's hosted `chroma-cloud-splade`
  with the same open model (`prithivida/Splade_PP_en_v1`), and keeps the rest of the source
  schema (sparse indexes were dropped when the Qwen function was replaced).
- Fixed: `pgchroma migrate` / `verify` reported a false failure ("check the collection's distance
  space") for collections with duplicate documents. Results that are as near as the exact k-th
  neighbour now count as exact.
- Changed: finished migration progress bars show the elapsed time instead of `0:00:00` remaining.
- Docs: README banner, demo recording, architecture diagram and benchmark chart
  (`docs/assets/render.py` regenerates them).

## 0.1.0 (2026-10-26)

First release.

- Chroma v2 HTTP API on Postgres + pgvector: tenants, databases, collections, `add`, `upsert`,
  `update`, `delete`, `get`, `query`, `count`, `fork`, `indexing_status`, token auth, base64
  embedding transport.
- Chroma's filter semantics (`where`, `where_document`), distances (`l2`, `cosine`, `ip`) and
  write semantics, pinned by 75 differential tests against a real Chroma 1.5.9 server and a
  JavaScript client test.
- Search API: `Knn`, `Rrf`, ranking arithmetic, `select`, `#id` / `#document` filters (dense
  vectors).
- `pgchroma migrate`: read-only, resumable, verified copy from Chroma Cloud or any Chroma server;
  `--local-embedding-functions` swaps `chroma-cloud-qwen` for the same open model run locally.
- HNSW indexes built in the background when searches need them (`halfvec` above 2,000 dims),
  exact search for small collections, iterative index scans for filtered searches on
  pgvector 0.8.
- `pgchroma doctor`, Docker image, docker compose file.
