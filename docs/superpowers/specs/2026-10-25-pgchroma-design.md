# pgchroma — design

Date: 2026-10-25. Status: approved for build (the owner delegated every decision; calls made
on their behalf are marked **[call]**).

## Why

On 2026-10-23 Chroma announced it is joining Replit and winding down Chroma Cloud:
writes stop on **2026-11-22**, reads stop on **2026-12-22**. Every Chroma Cloud user has to
move. Today's options:

- self-host open-source Chroma (one more stateful service to run, back up and scale;
  single-node Chroma has no Search API, no sparse vectors, no BM25), or
- move to another vector database and **rewrite the retrieval layer**.

Many of those users already run Postgres (Supabase, Neon, RDS, Railway...). pgchroma lets
them keep every line of Chroma client code and keep their vectors in that Postgres.

Research (2026-10-25) found many generic vector-migration CLIs and Chroma-to-X copy tools,
but no server that speaks Chroma's HTTP API on top of Postgres. That gap is the product.

## What it is

A stateless HTTP server that implements Chroma's v2 REST API (the one used by
`chromadb` >= 1.0 for Python and JS, LangChain, LlamaIndex, ...) and stores everything in
Postgres with pgvector, plus a `migrate` command that copies a Chroma deployment (Cloud or
self-hosted) into it and verifies the copy.

```
your app (unchanged chromadb client)  ──HTTP──▶  pgchroma serve  ──SQL──▶  Postgres + pgvector
Chroma Cloud / Chroma server  ──read-only──▶  pgchroma migrate  ──▶  same Postgres
```

## Goals

1. Drop-in: the official `chromadb` Python client (1.x) works unchanged against pgchroma for
   tenants, databases, collections, add / upsert / update / delete / get / query / count,
   metadata and document filters, and collection configuration round-trips.
2. Compatibility is proven, not claimed: a differential test suite runs the same client calls
   against a real Chroma server and against pgchroma and compares the results.
3. Stateless: any number of pgchroma processes can serve one database. All state is in
   Postgres. Deploy it like any web app.
4. One-command migration from Chroma Cloud with resume and verification (counts, record
   equality on samples, top-k query parity on samples). Read-only against the source, so it
   keeps working after Chroma Cloud turns read-only on Nov 22.
5. Fast enough for real use: base64 embedding transport, batched inserts, HNSW indexes,
   exact search where it is cheap.

## Non-goals

- Not a fork of Chroma and not a general vector database.
- No embedding computation on the server (Chroma does not do it either: clients embed).
- No Chroma Cloud-only features beyond the Search API subset listed below in v1:
  attached functions, collection forking across tenants, quotas, billing.
- Not a Postgres extension. It must work on managed Postgres where only `vector` is
  available.

## Anti-goals (would make it a failure even if it "works")

- Silent data changes: wrong distances, dropped metadata keys, int/float confusion,
  reordered results without saying so.
- Requiring users to change client code beyond the host/port/headers.
- Losing data during migration, or touching the source in any way other than reads.

## Constraints

- Postgres 14+ with pgvector 0.5+. Use pgvector 0.8 features (iterative index scans) when
  available, fall back to exact search for filtered queries otherwise.
- Python 3.10+. **[call]** Python because the Chroma ecosystem is Python-first, the
  differential tests drive the real `chromadb` client, and `uvx pgchroma` is a one-liner.
- Dependencies stay small: starlette, uvicorn, psycopg 3 (+pool), pgvector, orjson, numpy,
  httpx (migrate), rich (CLI). **[call]** Starlette instead of FastAPI so large `add`
  payloads skip per-field model validation.
- License MIT. **[call]**

## Storage model

All objects live in one Postgres schema (default `pgchroma`, configurable).

- `tenants(name pk, resource_name, created_at)`
- `databases(id uuid pk, tenant fk, name, created_at, unique(tenant, name))`; the default
  database uses id `00000000-0000-0000-0000-000000000000` like Chroma.
- `collections(id uuid pk, database_id fk, name, metadata jsonb, configuration jsonb,
  schema jsonb, dimension int, space text, version int, log_position bigint,
  created_at, unique(database_id, name))`
- One table per collection, `"c_<uuid hex>"`:
  `(seq bigint identity, id text primary key, embedding vector(dim), document text,
  metadata jsonb, uri text)`. **[call]** Table-per-collection instead of one big table:
  each collection gets an HNSW index with its own dimension and operator class, deletes of
  whole collections are `DROP TABLE`, and the planner never mixes collections.
  The table is created when the collection is created; the `embedding` column is typed to
  `vector(dim)` on the first write that establishes the dimension.
- `seq` reproduces Chroma's insertion order: `get` orders by it; upserting an existing id
  keeps its position.

Indexes: primary key on `id`, GIN (`jsonb_path_ops`) on `metadata`, HNSW on `embedding`
with `vector_l2_ops` / `vector_cosine_ops` / `vector_ip_ops` from the collection's space.
**[call]** The HNSW index is built once a collection passes `PGCHROMA_INDEX_THRESHOLD` rows
(default 10,000), in the background with `CREATE INDEX CONCURRENTLY`; below that, exact
scans are fast and return exact results.

## Semantics (observed on Chroma 1.5.9 and enforced by tests)

- `add` with an id that already exists is ignored without error.
- `update` of a missing id is ignored. Metadata updates merge: given keys overwrite, keys
  set to `null` are removed.
- `upsert` inserts or updates; an existing record keeps its position.
- `get(ids=...)` returns matches in storage order; unknown ids are skipped.
- `$ne` and `$nin` match records that do not have the key.
- Distances: `l2` is squared Euclidean, `cosine` is `1 - cos`, `ip` is `1 - dot`.
- Dimension is fixed by the first write; a mismatch is `InvalidArgumentError:
  Collection expecting embedding with dimension of X, got Y`.
- Errors use Chroma's body `{"error": "<Name>", "message": "..."}` and status codes, so the
  client raises the same exception classes.
- Embeddings may arrive as base64 little-endian float32 strings; responses use JSON floats.

## Filters

`where` compiles to SQL over `metadata jsonb` with type-aware comparisons (numbers compare
numerically, strings lexically, bools only equal bools), `$and/$or`, `$in/$nin`, and
`$contains/$not_contains` on list values. `where_document` supports `$contains`,
`$not_contains`, `$regex`, `$not_regex`, `$and`, `$or`. Regexes run as Postgres regexes;
the README lists the syntax differences from Rust's regex crate.

## Search API

Chroma Cloud's `/search` endpoint (the `Search()` builder) is a reason people cannot move
to single-node Chroma. v1 supports: `Knn` ranking on the dense embedding, `where` filters,
`limit/offset`, `select`. Hybrid (`Rrf`) with sparse vectors needs pgvector >= 0.7
(`sparsevec`) and is a stretch goal for this release.

## CLI

- `pgchroma serve --database-url ... [--host --port --schema --auth-token ...]`
- `pgchroma migrate --from <url|cloud> [--api-key --tenant --database --collection ...]
  --database-url ...` with resume and `--verify`
- `pgchroma verify` — re-run verification against the source.
- `pgchroma doctor` — check Postgres version, pgvector version, permissions, indexes.

## Auth

Off by default (like self-hosted Chroma). With `--auth-token` (or `PGCHROMA_AUTH_TOKENS`,
comma separated) every request must carry `x-chroma-token: <token>` or
`Authorization: Bearer <token>`, which is what Chroma clients send.

## Testing

- Unit tests for filter compilation and vector encoding.
- Integration tests against Postgres: `DATABASE_URL` if set (CI: `pgvector/pgvector:pg17`),
  otherwise an embedded server from `pgserver`.
- Differential tests: start the real Chroma server (`chroma run` from the `chromadb` wheel)
  and pgchroma, drive both with the official client, compare normalized results.
- Migration tests: real Chroma → pgchroma, then verify.

## Left to the builder

Exact module boundaries, logging format, progress UI, Docker base image.
