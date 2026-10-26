---
title: Chroma API compatibility
description: Which parts of Chroma's v2 HTTP API pgchroma implements, how it is tested against a real Chroma server, and the known differences.
---

# Compatibility

Tested with the official clients `chromadb` 1.5.9 (Python) and `chromadb` 3.5.0 (npm), against
a real Chroma 1.5.9 server, on Postgres 16 and 17 with pgvector 0.6, 0.7 and 0.8, in CI on every
push (the core suite also runs on Postgres 15).

| | |
|---|---|
| Tenants, databases, collections (create, get, get-or-create, list, count, modify, delete, fork) | ✅ |
| `add`, `upsert`, `update`, `delete`, `get`, `query`, `count`, `peek` | ✅ |
| `where`: `$eq $ne $gt $gte $lt $lte $in $nin $and $or`, list `$contains / $not_contains` | ✅ same results as Chroma, including its int/float rules |
| `where_document`: `$contains $not_contains $regex $not_regex $and $or` | ✅ |
| `include`, `limit` / `offset`, querying by `ids` | ✅ |
| Distance spaces `l2`, `cosine`, `ip` (same distance values as Chroma) | ✅ |
| Collection configuration, embedding-function config, metadata, legacy `hnsw:space` metadata | ✅ |
| Base64 embedding transport, max batch size, `pre-flight-checks` | ✅ |
| Token auth (`x-chroma-token` or `Authorization: Bearer`) | ✅ |
| Search API: `Search().where().rank(Knn / Rrf / arithmetic).limit().select()`, `#id` / `#document` filters | ✅ |
| Sparse vectors (`SparseVector` metadata), sparse `Knn`, BM25 weighting (`bm25=True`), hybrid `Rrf` of dense + sparse | ✅ scored like Chroma Cloud (`1 - query·document`, BM25 IDF over the collection), following Chroma's source; no self-hosted Chroma can run these to compare against |
| Search API: `group_by` with `MinK` / `MaxK` (top k per group) | ✅ following Chroma's source |
| Attached functions, quotas, Chroma Cloud billing APIs | ❌ |
| Pre-1.0 clients (v1 API) | ❌ answers `410` with an upgrade hint |

## Known differences

- `delete(ids=[...])` returns the **exact** number of records deleted. Chroma returns
  `len(ids)` even for ids that do not exist.
- `$regex` uses Postgres regular expressions instead of Rust's `regex` crate. Common syntax
  (classes, anchors, alternation, `(?i)`) behaves the same; exotic features may not.
- Approximate search is pgvector's HNSW, not Chroma's HNSW or SPANN, so for large collections the
  last few of the top-k results can differ, the same way two runs of two different ANN indexes
  differ. Small collections are searched exactly.
- Chroma Cloud collections configured for SPANN are served with HNSW.
- A float metadata value of 1e16 or more is written by Python in exponent notation and comes
  back from Postgres as an integer.
