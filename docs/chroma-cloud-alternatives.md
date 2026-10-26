---
title: Chroma Cloud alternatives after the shutdown
description: Where to move a Chroma Cloud app before Nov 22, 2026 - self-hosted Chroma, Chroma's distributed deployment, Postgres with pgchroma, other Chroma-compatible servers, or a managed vector database - and what each one costs you.
---

# Chroma Cloud alternatives

Chroma Cloud stops writes on November 22 and reads on December 22, 2026. There are five
realistic places to go. The right one depends on two questions: do you want to keep your
`chromadb` code, and do you use sparse vectors, BM25 or the Search API?

| | Keeps your chromadb code | Sparse / BM25 / Search API | What you run |
|---|---|---|---|
| Self-hosted Chroma (single node) | Yes | No | One container or `pip install chromadb` |
| Chroma distributed | Yes | Yes | Kubernetes, object storage, several services |
| **pgchroma** on Postgres | Yes | Yes | One stateless container + any Postgres with pgvector |
| Another Chroma-compatible server | Yes | Depends on the server | That server and its storage |
| Managed vector database (Pinecone, etc.) | No, new client and data model | Depends on the product | Nothing; you pay per use |

## Self-hosted Chroma

Open-source Chroma continues under Apache 2.0, and Chroma's own CLI copies collections down:
`chroma copy --from-cloud --all --db my-db`. If you only use dense vectors with metadata and
document filters, this is the smallest change.

What it doesn't do: single-node Chroma answers "Search operation is not implemented for local
executor" for the Search API and rejects sparse vectors, so collections that use BM25, SPLADE or
hybrid search can't move there. You also take on running and backing up another stateful
service.

## Chroma distributed

The deployment behind Chroma Cloud is open source too. It supports everything Chroma Cloud did,
but it is a distributed system: Kubernetes, object storage and several services to operate.
Worth it at large scale, heavy for most teams.

## pgchroma on Postgres

pgchroma speaks Chroma's HTTP API and stores collections in Postgres with pgvector. Your
`chromadb` code (Python, JavaScript, LangChain, LlamaIndex) keeps working, and so do the Search
API, sparse vectors and hybrid search. One command copies Chroma Cloud and verifies the copy.

Choose it when you already run Postgres (Supabase, Neon, RDS, Cloud SQL) and would rather not
add another database: vectors become tables you can back up, replicate and join with the rest of
your data. [Migration guide](migrate-from-chroma-cloud.md).

Trade-offs: plain vector search is a little slower than Chroma's (7.6 vs 5.3 ms p50 in our
[benchmark](index.md#performance)), sparse search is exact rather than indexed, and pgchroma is
new software, so verify before you switch.

## Other Chroma-compatible servers

Other independent projects also implement Chroma's API; for example
[VecturaDB](https://vecturadb.org/leaving-chroma-cloud/) publishes migration guides tested
against Chroma Cloud. Compare what they store your data in and how they are operated.

## A managed vector database

Pinecone and others offer managed services and migration guides
([Pinecone's](https://www.pinecone.io/blog/migrate-chroma-cloud-to-pinecone/)). You stop operating
anything, but you rewrite your retrieval code for a new client and data model, re-test search
quality, and accept another vendor dependency, which is the situation Chroma Cloud users are in now.

## Rewrite on pgvector directly

If you want Postgres without a Chroma-compatible layer, you can use pgvector from your own SQL.
You then rewrite your retrieval code, filters and integrations yourself; pgchroma does that
translation for you and leaves the same tables underneath.
