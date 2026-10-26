---
title: Deploy pgchroma on Supabase, Neon, RDS or Docker
description: Run pgchroma, the Chroma-compatible server for Postgres, against Supabase, Neon, AWS RDS, Cloud SQL, Azure or any Postgres with pgvector. Connection strings, poolers and settings.
---

# Deploy

pgchroma is a stateless web app: give it `DATABASE_URL` and a port. It creates a `pgchroma`
schema with its tables on first start and touches nothing else in the database.

```bash
docker run -p 8000:8000 -e DATABASE_URL="postgresql://user:pass@host:5432/db" ghcr.io/thejustas228/pgchroma
```

The image (amd64 and arm64) listens on `$PORT`, so it also runs as-is on Railway, Render,
Fly.io, Cloud Run and similar platforms.

## Neon

Tested on Neon (Postgres 17, pgvector 0.8): the test suite passes over the direct connection,
and the API works over the pooled one.

1. Copy either connection string. pgchroma recognises Neon's pooled endpoint (the host ends in
   `-pooler`) and adapts to it.
2. `pgchroma serve --database-url "$DATABASE_URL"`.

pgchroma creates the `vector` extension itself if the role is allowed to; otherwise run
`CREATE EXTENSION vector;` once in the SQL editor.

## Supabase

Enable the `vector` extension (Database → Extensions), then use any of the three connection
strings:

- **Direct connection** (`db.<project>.supabase.co:5432`): fastest; IPv6 unless you have the
  IPv4 add-on.
- **Session pooler** (`pooler.supabase.com:5432`): works everywhere.
- **Transaction pooler** (`pooler.supabase.com:6543`): supported; pgchroma detects port 6543.
  (Transaction pooling is tested against PgBouncer in CI and Neon's pooler; Supavisor uses the
  same mode.)

## AWS RDS, Aurora, Cloud SQL, Azure

pgvector is available on all of them. Run `CREATE EXTENSION vector;` once as an admin, then point
`DATABASE_URL` at the database. pgvector 0.8 or newer is recommended: filtered searches use its
iterative index scans.

## Transaction poolers

Behind PgBouncer or Supavisor in transaction mode, consecutive statements can run on different
server connections, so session settings don't stick. pgchroma then sets its search path inside
every transaction and coordinates index builds through a table instead of a session lock.

It switches to this mode automatically for Neon's `-pooler` hosts and port 6543, and on its own
the first time it sees its tables disappear mid-session. For any other PgBouncer, start in that
mode with `PGCHROMA_POOLER=transaction`. Behind a pooler, index builds use the server's default
`maintenance_work_mem` instead of raising it.

## Settings

| Variable / flag | Default | |
|---|---|---|
| `DATABASE_URL` / `--database-url` | required | Postgres connection string |
| `PORT` / `--port`, `HOST` / `--host` | `8000`, `0.0.0.0` | listen address |
| `PGCHROMA_SCHEMA` / `--schema` | `pgchroma` | schema for pgchroma's tables |
| `PGCHROMA_AUTH_TOKENS` / `--auth-token` | off | comma-separated tokens; clients send `x-chroma-token` |
| `PGCHROMA_INDEX_THRESHOLD` / `--index-threshold` | `10000` | records before a collection gets an HNSW index |
| `PGCHROMA_POOL_SIZE` / `--pool-size` | `10` | Postgres connections per worker |
| `PGCHROMA_POOLER` | auto | `transaction` when running behind a transaction pooler |
| `PGCHROMA_MAINTENANCE_WORK_MEM` | `1GB` | memory for index builds (direct connections) |
| `--workers` | `1` | worker processes |

`pgchroma doctor` shows the Postgres and pgvector versions and every collection with its size
and index state.

## Auth

```bash
pgchroma serve --auth-token "$TOKEN"
```

```python
client = chromadb.HttpClient(host="...", port=8000, headers={"x-chroma-token": TOKEN})
```
