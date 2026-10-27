---
title: Export your Chroma Cloud data to files before it shuts down
description: Back up every Chroma Cloud collection (ids, embeddings, documents, metadata) to JSON Lines files with one command, before Chroma Cloud stops serving reads on December 22, 2026. Works during the read-only month.
---

# Export your Chroma Cloud data to files

Chroma Cloud stops accepting writes on **November 22, 2026** and stops serving reads on
**December 22, 2026** ([Chroma's notice](https://www.trychroma.com/cloud-wind-down)). After
that, whatever is still only in Chroma Cloud is gone. If you haven't decided where your vectors
should live yet, save them to files now and decide later.

```bash
export CHROMA_API_KEY="ck-..."

uvx --from git+https://github.com/TheJustas228/pgchroma pgchroma export \
    --from-cloud --api-key "$CHROMA_API_KEY" --out chroma-backup
```

You need [uv](https://docs.astral.sh/uv/) (or `pip install git+https://github.com/TheJustas228/pgchroma`).
No database, no account, nothing else.

- **Everything in the account:** every database and collection the API key can see, with ids,
  embeddings, documents, metadata and uris. Pick some with `--database` and `--collection`
  (both repeatable).
- **Read-only:** it only reads from Chroma Cloud, so it also works between November 22 and
  December 22, when Chroma Cloud is read-only.
- **Resumable:** if the network drops or you stop it, run the same command again. It keeps what
  it saved, drops a half-written last line, and continues where it stopped.
- **Self-hosted Chroma too:** `--from http://my-chroma:8000` instead of `--from-cloud`.

## What you get

```text
chroma-backup/
  manifest.json                    # source, tenant, and which collections are complete
  default_database/
    support-tickets.jsonl
    product-docs.jsonl
```

Each collection is one [JSON Lines](https://jsonlines.org/) file. The first line describes the
collection exactly as Chroma lists it (name, id, metadata, configuration, schema), and every
other line is one record:

```json
{"collection": {"id": "6f1c…", "name": "product-docs", "metadata": {"team": "docs"}, "configuration_json": {"hnsw": {"space": "cosine"}, …}, …}}
{"id": "doc-1", "embedding": [0.0342, -0.511, …], "document": "Refunds take 5 days…", "metadata": {"lang": "en"}, "uri": null}
```

It's plain text, so anything that reads JSON can use it:

```python
import json

with open("chroma-backup/default_database/product-docs.jsonl") as f:
    collection = json.loads(next(f))["collection"]  # name, metadata, configuration
    for line in f:
        record = json.loads(line)  # id, embedding, document, metadata, uri
```

Embeddings stored as text take space: about 18 KB per record at 1,536 dimensions, so roughly
1.8 GB per 100,000 records. Compress the folder if you're keeping it.

## Load it into Postgres later

The export is also a migration source. When you're ready, load it into Postgres (with
pgvector) and serve it with the same Chroma API your code already uses:

```bash
pgchroma migrate --from chroma-backup --database-url "$DATABASE_URL"
pgchroma serve --database-url "$DATABASE_URL"
```

`migrate` checks the result against the files the same way it checks a direct copy: record
counts, a random sample compared field by field, and nearest-neighbour searches. Then continue
with [step 3 of the migration guide](migrate-from-chroma-cloud.md#3-serve-the-chroma-api-from-postgres).

If you'd rather go to self-hosted Chroma, Chroma's own `chroma copy --from-cloud` copies
straight from Chroma Cloud to a Chroma server; keeping this export as a backup still makes sense.

## Hosted embedding functions

Collections that use Chroma Cloud's hosted embedding models (`chroma-cloud-qwen`,
`chroma-cloud-splade`) keep their stored embeddings in the export; only *new* embeddings need a
model. `pgchroma export` warns about these collections, and
`pgchroma migrate --from chroma-backup --local-embedding-functions` sets up the same open models
to run locally. See [hosted embedding functions](embedding-functions.md).
