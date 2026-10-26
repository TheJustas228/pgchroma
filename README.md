<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="pgchroma: Chroma's API. Your Postgres." src="docs/assets/banner-light.svg" width="100%">
</picture>

pgchroma is a server that speaks [Chroma](https://www.trychroma.com/)'s HTTP API and keeps every
collection in Postgres with [pgvector](https://github.com/pgvector/pgvector). Point your existing
Chroma client at it, keep your code, and keep your vectors next to the rest of your data:
Supabase, Neon, RDS, Cloud SQL, Railway, or the Postgres on your laptop.

[![CI](https://github.com/TheJustas228/pgchroma/actions/workflows/ci.yml/badge.svg)](https://github.com/TheJustas228/pgchroma/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Chroma API](https://img.shields.io/badge/Chroma%20API-v2%20(chromadb%201.x)-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-thejustas228.github.io%2Fpgchroma-indigo)](https://thejustas228.github.io/pgchroma/)

> **Leaving Chroma Cloud?** Chroma Cloud stops accepting writes on **November 22, 2026** and
> stops serving reads on **December 22, 2026**
> ([announcement](https://www.trychroma.com/cloud-wind-down)). `pgchroma migrate` copies
> your collections into Postgres, verifies them, and your code keeps working with a one-line
> change. [Jump to the migration guide.](#move-off-chroma-cloud-in-three-commands)

<img alt="Terminal recording: pgchroma migrate copies three collections (16,200 records) from a Chroma server into Postgres and verifies them, then pgchroma serve answers a query from unchanged chromadb code" src="docs/assets/demo.gif" width="100%">

<sub>A real run: three collections copied from a local Chroma 1.5.9 server and queried through
pgchroma with the stock <code>chromadb</code> client. <code>--from-cloud</code> does the same
against Chroma Cloud.</sub>

```python
import chromadb

# before: client = chromadb.CloudClient(api_key="ck-...", tenant="...", database="prod")
client = chromadb.HttpClient(host="localhost", port=8000, database="prod")

docs = client.get_collection("docs")  # everything below is unchanged
docs.query(query_texts=["how do refunds work?"], n_results=5, where={"lang": "en"})
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Your app's chromadb client talks the Chroma HTTP API to pgchroma serve, which stores everything in Postgres + pgvector. pgchroma migrate reads from Chroma Cloud or a self-hosted Chroma and writes to the same Postgres." src="docs/assets/architecture-light.svg" width="100%">
</picture>

## Why

- **No rewrite.** The official `chromadb` clients (Python and JavaScript) and everything built
  on them work unchanged. Same collections, filters, includes, distances and errors.
- **Proven compatible, not claimed.** The test suite drives the official client against a real
  Chroma 1.5.9 server and against pgchroma with the same calls, and compares the results:
  75 differential scenarios plus the JS client. [See what is covered.](#compatibility)
- **Stateless.** pgchroma keeps nothing on disk. Run one or twenty instances behind a load
  balancer. Backups, replication, point-in-time recovery and access control are Postgres's.
- **One database.** Join your embeddings with your users, orders and permissions in SQL.
  Your vectors are just tables: `pgchroma.c_<collection id>`.
- **Things single-node Chroma can't do.** The Chroma Cloud **Search API** (`Search`, `Knn`,
  `Rrf`, ranking expressions) and **sparse vectors** (BM25, SPLADE, hybrid search) work on
  pgchroma. Single-node Chroma answers "not implemented" for the Search API and rejects sparse
  vectors; pgchroma runs both on the Postgres you already have, with no Kubernetes cluster.

## Quickstart

You need Postgres with the `vector` extension available (Supabase, Neon, RDS, Cloud SQL,
Azure, Railway and most managed providers ship it; locally use the Docker image below).

**With Docker Compose** (Postgres 17 + pgvector + pgchroma):

```bash
git clone https://github.com/TheJustas228/pgchroma && cd pgchroma
docker compose up
```

**With Docker, against your own Postgres:**

```bash
docker run -p 8000:8000 -e DATABASE_URL="postgresql://user:pass@host:5432/db" ghcr.io/thejustas228/pgchroma
```

**Against your own Postgres** (Python 3.10+, [uv](https://docs.astral.sh/uv/)):

```bash
uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma serve \
    --database-url "postgresql://user:pass@host:5432/db"
```

Then use Chroma exactly as before:

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
notes = client.get_or_create_collection("notes", configuration={"hnsw": {"space": "cosine"}})
notes.add(ids=["a", "b"], documents=["pgchroma stores vectors in Postgres", "cats sleep a lot"])
print(notes.query(query_texts=["where are my embeddings?"], n_results=1)["documents"])
```

```js
import { ChromaClient } from "chromadb";
const client = new ChromaClient({ host: "localhost", port: 8000 });
```

pgchroma creates a `pgchroma` schema (configurable with `--schema`) and its tables on first
start. It never touches anything else in the database.

## Move off Chroma Cloud in three commands

```bash
# 1. copy everything (read-only on Chroma Cloud, resumable, verified at the end)
uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma migrate \
    --from-cloud --api-key "$CHROMA_API_KEY" \
    --database-url "$DATABASE_URL"

# 2. serve the Chroma API from Postgres
uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma serve --database-url "$DATABASE_URL"

# 3. swap CloudClient for HttpClient in your app (the migrate output prints the exact line)
```

The same migration as text (three collections with real embeddings from Chroma's default
model):

```text
Source: http://localhost:8001 tenant default_tenant
user-memories   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 800/800     0:00:00
support-tickets ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12000/12000 0:00:03
product-docs    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3400/3400   0:00:01
Migrated
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃ collection      ┃ from database    ┃ to database      ┃ records ┃ seconds ┃ index ┃ resumed ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│ user-memories   │ default_database │ default_database │     800 │     0.4 │ exact │         │
│ support-tickets │ default_database │ default_database │  12,000 │     5.1 │  hnsw │         │
│ product-docs    │ default_database │ default_database │   3,400 │     1.3 │ exact │         │
└─────────────────┴──────────────────┴──────────────────┴─────────┴─────────┴───────┴─────────┘
Copied 16,200 records in 5.1s.
Verification
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ collection      ┃ source ┃ target ┃ sampled ┃ differ ┃ top-k vs exact ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ user-memories   │    800 │    800 │      48 │      0 │           100% │
│ support-tickets │  12000 │  12000 │      48 │      0 │           100% │
│ product-docs    │   3400 │   3400 │      48 │      0 │           100% │
└─────────────────┴────────┴────────┴─────────┴────────┴────────────────┘
✔ Verified: counts match and every sampled record is identical.

Next: start the server (pgchroma serve) and point your client at it:
  chromadb.HttpClient(host="localhost", port=8000, tenant="default_tenant", database="default_database")
```

Details that matter:

- **It only reads from the source**, so it also works during Chroma Cloud's read-only month
  (Nov 22 – Dec 22).
- **Resumable.** Progress is checkpointed per batch in Postgres. If the network drops, run the
  same command again; it continues where it stopped and never duplicates records.
- **Verified.** Counts, a random sample of records (ids, embeddings, documents, metadata, uris)
  and query parity against exact search are checked for every collection. A source that uses an
  approximate index (Chroma's HNSW or SPANN) may miss a few true neighbours on large
  collections; below 90% agreement is reported as a warning, below 50% as a failure.
  `pgchroma verify` re-runs the checks later.
- **Where things land.** Collections keep their names, metadata and configuration. They go into
  tenant `default_tenant` and a database with the same name as in Chroma Cloud. Change that
  with `--to-tenant` / `--to-database`. Pick collections with `--collection` and databases with
  `--database` (both repeatable).
- **Indexes are built once, after the copy**, which is much faster than maintaining them during
  inserts.
- **Self-hosted Chroma works too:** `--from http://my-chroma:8000`.

### Chroma Cloud's hosted embedding functions

The embeddings you already stored are copied as they are. Only *new* embeddings (for `add` with
`documents` and for `query_texts`) need a model, and Chroma Cloud's hosted ones stop with Chroma
Cloud. `chroma-cloud-qwen` and `chroma-cloud-splade` are open models
(`Qwen/Qwen3-Embedding-0.6B` and `prithivida/Splade_PP_en_v1`), so you can run them yourself:

```bash
pip install sentence-transformers
pgchroma migrate --from-cloud --api-key "$CHROMA_API_KEY" --database-url "$DATABASE_URL" \
    --local-embedding-functions
```

With `--local-embedding-functions`, collections that used `chroma-cloud-qwen` (dense) or
`chroma-cloud-splade` (sparse) are configured to use the same open models
(`Qwen/Qwen3-Embedding-0.6B`, `prithivida/Splade_PP_en_v1`) through sentence-transformers. The client loads it from the collection
configuration, so `query_texts=` and `add(documents=...)` keep working without code changes. If
your collection used a Qwen *task* (such as `nl_to_code`), queries need an instruction prefix;
[`examples/local_qwen_embeddings.py`](examples/local_qwen_embeddings.py) shows how. Without
the flag, `pgchroma migrate` warns about every collection that uses a hosted function.

## Compatibility

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

### Known differences

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

## Performance

`bench/bench.py`, same client and data against both servers: 50,000 × 384-dim cosine vectors,
batches of 1,000, 200 queries, `n_results=10`. One machine (16 cores, WSL2), Chroma 1.5.9 vs
pgchroma on Postgres 16 + pgvector 0.6.2 with default settings. A sanity check, not a
leaderboard.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-dark.svg">
  <img alt="Bar charts, pgchroma vs Chroma 1.5.9: ingest 12,497 vs 1,274 records/s; filtered query p50 11.4 vs 37.2 ms; get(where) p50 3.9 vs 11.3 ms; query p50 7.6 vs 5.3 ms (Chroma is faster here)" src="docs/assets/benchmark-light.svg" width="100%">
</picture>

| | ingest | query p50 | filtered query p50 | `get(where=...)` p50 |
|---|---:|---:|---:|---:|
| Chroma 1.5.9 | 1,274 rec/s | **5.3 ms** | 37.2 ms | 11.3 ms |
| pgchroma | **12,497 rec/s** | 7.6 ms | **11.4 ms** | **3.9 ms** |

How it gets there:

- Embeddings travel base64-encoded from the client and binary to Postgres.
- Small collections (below 10,000 records, `--index-threshold`) are searched exactly with a
  sequential scan.
- Larger collections get an HNSW index with the collection's `max_neighbors` and
  `ef_construction`, built in the background with `CREATE INDEX CONCURRENTLY` the first time a
  search needs it, never during bulk writes. `collection.get_indexing_status()` reports
  progress. Embeddings above 2,000 dimensions are indexed as `halfvec` (pgvector ≥ 0.7).
- Metadata has a GIN index; equality filters use it.
- Filtered searches on pgvector ≥ 0.8 use iterative index scans, so a selective filter still
  returns `n_results` hits. On older pgvector, filtered searches run exact.
- Sparse (BM25/SPLADE) searches are exact: candidates come from the jsonb index, then every
  candidate is scored. 12,000 support tickets with BM25 vectors answer in about 60 ms; there is
  no inverted index yet, so very large sparse collections are slower than on Chroma Cloud.

## Deploy

pgchroma is a stateless web app. Give it `DATABASE_URL` and a port.

| Variable / flag | Default | |
|---|---|---|
| `DATABASE_URL` / `--database-url` | required | Postgres connection string |
| `PORT` / `--port`, `HOST` / `--host` | `8000`, `0.0.0.0` | listen address |
| `PGCHROMA_SCHEMA` / `--schema` | `pgchroma` | schema for pgchroma's tables |
| `PGCHROMA_AUTH_TOKENS` / `--auth-token` | off | comma-separated tokens; clients send `x-chroma-token` |
| `PGCHROMA_INDEX_THRESHOLD` / `--index-threshold` | `10000` | records before a collection gets an HNSW index |
| `PGCHROMA_POOL_SIZE` / `--pool-size` | `10` | Postgres connections per worker |
| `--workers` | `1` | worker processes |
| `PGCHROMA_POOLER` | auto | `transaction` behind PgBouncer in transaction mode (Neon and Supabase poolers are detected) |
| `PGCHROMA_MAINTENANCE_WORK_MEM` | `1GB` | memory for index builds (session setting, direct connections only) |

- **Docker:** `ghcr.io/thejustas228/pgchroma` (amd64 and arm64). The image listens on `$PORT`,
  so it runs as-is on Railway, Render, Fly.io, Cloud Run and similar platforms.
- **Neon:** tested (Postgres 17, pgvector 0.8). Either connection string works; the pooled
  `-pooler` endpoint is detected automatically.
- **Supabase:** enable the `vector` extension; the direct connection, the session pooler and the
  transaction pooler (port 6543, detected automatically) all work.
- **Other transaction poolers (PgBouncer):** set `PGCHROMA_POOLER=transaction`. pgchroma then
  sets its search path in every transaction and coordinates index builds through a table.
- **RDS / Cloud SQL / Azure:** enable `vector` once (`CREATE EXTENSION vector;`), then point
  `DATABASE_URL` at the database.
- **Auth:** set `PGCHROMA_AUTH_TOKENS`, then in Python
  `chromadb.HttpClient(host=..., headers={"x-chroma-token": "<token>"})`.

`pgchroma doctor` checks the Postgres and pgvector versions and shows every collection with its
size and index state.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py): plain `chromadb` client code against pgchroma
- [`examples/langchain_rag.py`](examples/langchain_rag.py): the stock `langchain-chroma` vector store and retriever
- [`examples/local_qwen_embeddings.py`](examples/local_qwen_embeddings.py): Chroma Cloud's Qwen embeddings, locally
- [`bench/bench.py`](bench/bench.py): the benchmark above

## How it works

```text
pgchroma.tenants      (name)
pgchroma.databases    (id, tenant, name)
pgchroma.collections  (id, database_id, name, metadata, configuration, schema, dimension, space)
pgchroma.c_<uuid>     (seq, id, embedding vector(dim), document, metadata jsonb, uri)   -- one per collection
```

- One table per collection, so every collection gets an index with its own dimension and
  distance operator, and deleting a collection is a `DROP TABLE`.
- `where` / `where_document` filters compile to parameterized SQL over `jsonb`, matching
  Chroma's semantics (including that `$ne` matches records without the key, and Chroma's
  int/float comparison rules).
- `query` runs one `LATERAL` nearest-neighbour search per query embedding, in one statement.
- Writes follow Chroma: `add` ignores existing ids, `update` and `upsert` merge metadata
  (`None` deletes a key), and records keep their insertion order.

## Development

```bash
uv sync
uv run pytest            # uses $DATABASE_URL, or starts an embedded Postgres (pgserver)
```

The differential tests start a real Chroma server from the `chromadb` wheel for every test,
so they need Python 3.11+. `tests/test_js_client.py` needs Node.js.

## FAQ

**Is this affiliated with Chroma?** No. pgchroma is an independent open-source project that
implements Chroma's public HTTP API. Chroma is a trademark of its owners.

**Why not just self-host Chroma?** You can, and for many people that's the right move:
Chroma's own `chroma copy --from-cloud` moves dense collections to a self-hosted Chroma.
pgchroma is for teams that would rather not run another stateful service, who already trust
Postgres with their data, who want SQL joins between vectors and the rest of their data, or who
use the Search API, which single-node Chroma doesn't implement.

**Why not use pgvector directly?** Then you rewrite your retrieval code, filters and
integrations. pgchroma keeps them and still gives you plain Postgres tables underneath.

**Is it production ready?** It's new. The behavior is pinned by a test suite that runs against
the real Chroma server, but give it the testing you'd give any new dependency, and keep your
Chroma Cloud data until you have verified the migration. Issues and PRs are welcome.

## License

MIT

If pgchroma saves you a migration, a ⭐ on the repo helps other Chroma users find it.
