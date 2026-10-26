# Changelog

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
