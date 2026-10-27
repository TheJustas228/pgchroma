---
title: How to migrate off Chroma Cloud to Postgres
description: Step-by-step guide to moving Chroma Cloud collections to Postgres + pgvector before Chroma Cloud stops writes on Nov 22 and reads on Dec 22, 2026, without rewriting your chromadb code.
---

# How to migrate off Chroma Cloud

Chroma Cloud stops accepting writes on **November 22, 2026** and stops serving reads on
**December 22, 2026** ([Chroma's notice](https://www.trychroma.com/cloud-wind-down)). This guide
moves every collection to Postgres with pgchroma and keeps your application code as it is.

**What you need:** a Chroma Cloud API key, a Postgres database with the `vector` extension
available (Supabase, Neon, RDS, Cloud SQL, Azure, Railway, or Docker), and Python 3.10+ with
[uv](https://docs.astral.sh/uv/) or Docker.

!!! tip "Not sure where your vectors should go yet?"
    [Export them to files first](export-chroma-cloud.md): one command, no database, and the
    export can be loaded into Postgres later with the same verification.

## 1. Copy your collections

```bash
export CHROMA_API_KEY="ck-..."
export DATABASE_URL="postgresql://user:pass@host:5432/db"

uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma migrate \
    --from-cloud --api-key "$CHROMA_API_KEY" --database-url "$DATABASE_URL"
```

This copies every database and collection your key can see: ids, embeddings, documents,
metadata (including sparse vectors), URIs, collection metadata and configuration.

- **Read-only on Chroma Cloud.** It never writes to the source, so it also works during the
  read-only month (Nov 22 to Dec 22).
- **Resumable.** Progress is checkpointed in Postgres per batch. If the network drops, run the
  same command again; it continues where it stopped and never duplicates records.
- **Quota-aware.** Rate limits are retried with backoff and batches shrink when Chroma Cloud
  rejects a request as too large.
- **Selective.** `--database prod` and `--collection docs` (both repeatable) pick what to copy.
  `--to-database` and `--to-tenant` choose where it lands.

## 2. Check the result

The command ends with a verification table for every collection: record counts on both sides,
a random sample of records compared field by field, and how many of Chroma Cloud's top-10 search
results are exact nearest neighbours in Postgres.

```text
Verification
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ collection      ┃ source ┃ target ┃ sampled ┃ differ ┃ top-k vs exact ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ user-memories   │    800 │    800 │      48 │      0 │           100% │
│ support-tickets │  12000 │  12000 │      48 │      0 │           100% │
│ product-docs    │   3400 │   3400 │      48 │      0 │           100% │
└─────────────────┴────────┴────────┴─────────┴────────┴────────────────┘
✔ Verified: counts match and every sampled record is identical.
```

Chroma Cloud's own index is approximate, so on large collections a few of its results may not
be exact neighbours; below 90% agreement is shown as a warning, below 50% as a failure.
`pgchroma verify` runs the same checks again later.

## 3. Serve the Chroma API from Postgres

```bash
uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma serve --database-url "$DATABASE_URL"
# or: docker run -p 8000:8000 -e DATABASE_URL="$DATABASE_URL" ghcr.io/thejustas228/pgchroma
```

## 4. Point your app at it

```python
# before
client = chromadb.CloudClient(api_key="ck-...", tenant="...", database="prod")
# after
client = chromadb.HttpClient(host="pgchroma.internal", port=8000, database="prod")
```

```js
// before: new CloudClient({ apiKey: "ck-...", tenant: "...", database: "prod" })
const client = new ChromaClient({ host: "pgchroma.internal", port: 8000, database: "prod" });
```

Collections keep their names. They land in tenant `default_tenant`, in a database with the same
name as on Chroma Cloud, so `database=` stays the same. To require a token like Chroma Cloud did,
start the server with `--auth-token` and send it as the `x-chroma-token` header.

## A safe cutover

1. Run `pgchroma migrate` once while your app still writes to Chroma Cloud. This copies the bulk
   of the data and tells you how long a full copy takes.
2. Pause writes (or queue them) and run `pgchroma migrate` again. It continues from its
   checkpoint, so records added since the first run are copied. That is enough for collections
   that only grow. If your app also updates or deletes records, migrate into a fresh database
   (`--to-database prod-final`) during the pause instead.
3. Check the verification table, switch reads and writes to pgchroma together, and keep Chroma
   Cloud untouched until December 22 in case you need to compare.

## Chroma Cloud's hosted embedding models

The embeddings you already have are copied as they are. Only new documents and `query_texts`
need a model, and Chroma Cloud's hosted ones stop with Chroma Cloud. Add
`--local-embedding-functions` and pgchroma configures the same open models to run locally;
see [hosted embedding functions](embedding-functions.md).

## Troubleshooting

- **`Collection [...] already exists`**: pgchroma never merges into a collection it did not
  create. Delete it, or migrate with `--to-database`.
- **"Collection does not exist" right after connecting through PgBouncer**: set
  `PGCHROMA_POOLER=transaction`. Neon's and Supabase's poolers are detected automatically. See
  [deploy](deploy.md#transaction-poolers).
- **Something differs from Chroma**: [open an issue](https://github.com/TheJustas228/pgchroma/issues/new/choose)
  with the client code; compatibility gaps are treated as bugs.
