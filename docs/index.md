---
title: Chroma's API on your Postgres
description: pgchroma is a Chroma-compatible server on Postgres + pgvector. Keep your chromadb code, move off Chroma Cloud with one verified command, and keep your vectors next to the rest of your data.
hide:
  - navigation
---

# pgchroma: Chroma's API on your Postgres { .visually-hidden }

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img alt="pgchroma: Chroma's API. Your Postgres." src="assets/banner-light.svg" width="100%">
</picture>

pgchroma is a server that speaks [Chroma](https://www.trychroma.com/)'s HTTP API and keeps every
collection in Postgres with [pgvector](https://github.com/pgvector/pgvector). Your existing Chroma
client code keeps working: you change the host, not the code.

[Move off Chroma Cloud](migrate-from-chroma-cloud.md){ .md-button .md-button--primary }
[Star on GitHub](https://github.com/TheJustas228/pgchroma){ .md-button }

!!! warning "Chroma Cloud is shutting down"
    Chroma Cloud stops accepting writes on **November 22, 2026** and stops serving reads on
    **December 22, 2026** ([announcement](https://www.trychroma.com/cloud-wind-down)).
    [Here is how to move your collections to Postgres.](migrate-from-chroma-cloud.md)

<img alt="Terminal recording: pgchroma migrate copies three collections from a Chroma server into Postgres and verifies them, then answers a query from unchanged chromadb code" src="assets/demo.gif" width="100%">

```python
import chromadb

# before: client = chromadb.CloudClient(api_key="ck-...", tenant="...", database="prod")
client = chromadb.HttpClient(host="localhost", port=8000, database="prod")

docs = client.get_collection("docs")  # everything below is unchanged
docs.query(query_texts=["how do refunds work?"], n_results=5, where={"lang": "en"})
```

## Why pgchroma

- **No rewrite.** The official `chromadb` clients for Python and JavaScript, and everything built
  on them (LangChain, LlamaIndex), work unchanged.
- **Tested against the real thing.** The test suite runs the official client against a real
  Chroma 1.5.9 server and against pgchroma with the same calls and compares the results: 75
  scenarios plus the JavaScript client.
- **Your Postgres.** Supabase, Neon, RDS, Cloud SQL, Azure, Railway, or a laptop. Backups,
  replicas, access control and SQL joins come with it.
- **More than single-node Chroma.** Chroma Cloud's Search API, sparse vectors (BM25, SPLADE) and
  hybrid search work on pgchroma. Single-node Chroma rejects them.
- **Stateless.** Run one instance or twenty behind a load balancer.

## Quickstart

```bash
docker run -p 8000:8000 -e DATABASE_URL="postgresql://user:pass@host:5432/db" ghcr.io/thejustas228/pgchroma
```

or, without Docker:

```bash
uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma serve --database-url "$DATABASE_URL"
```

Then point any Chroma client at `localhost:8000`.

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmark-dark.svg">
  <img alt="pgchroma vs Chroma 1.5.9 on 50,000 384-dimensional vectors: ingest 12,497 vs 1,274 records/s; filtered query 11.4 vs 37.2 ms; get with where 3.9 vs 11.3 ms; query 7.6 vs 5.3 ms" src="assets/benchmark-light.svg" width="100%">
</picture>

Measured with [`bench/bench.py`](https://github.com/TheJustas228/pgchroma/blob/main/bench/bench.py):
same client and data against both servers, one machine. Plain vector search is a little slower
than Chroma; ingest and filtered search are much faster.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
  <img alt="Your app's chromadb client talks the Chroma HTTP API to pgchroma serve, which stores everything in Postgres + pgvector. pgchroma migrate reads from Chroma Cloud and writes to the same Postgres." src="assets/architecture-light.svg" width="100%">
</picture>

Every collection is a Postgres table with a `vector` column, a `jsonb` metadata column and its own
HNSW index. Filters compile to SQL. Nothing is stored outside Postgres.

---

pgchroma is MIT licensed and not affiliated with Chroma. If it saves you a migration, a
[star on GitHub](https://github.com/TheJustas228/pgchroma) helps other Chroma users find it.
