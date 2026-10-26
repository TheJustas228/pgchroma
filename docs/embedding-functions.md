---
title: Replacing Chroma Cloud's hosted embedding models
description: chroma-cloud-qwen and chroma-cloud-splade stop with Chroma Cloud. They are open models (Qwen3-Embedding-0.6B, Splade_PP_en_v1) you can run locally; pgchroma migrate can configure that for you.
---

# Chroma Cloud's hosted embedding functions

Chroma Cloud ran two embedding models on its own servers. They stop when Chroma Cloud stops.

| Embedding function | Open model | Kind |
|---|---|---|
| `chroma-cloud-qwen` | `Qwen/Qwen3-Embedding-0.6B` | dense, 1024 dimensions |
| `chroma-cloud-splade` | `prithivida/Splade_PP_en_v1` | sparse |

**Your stored embeddings don't change.** `pgchroma migrate` copies them as they are. Only new
documents and query text need a model.

## Let the migration switch them

```bash
pip install sentence-transformers
pgchroma migrate --from-cloud --api-key "$CHROMA_API_KEY" --database-url "$DATABASE_URL" \
    --local-embedding-functions
```

Collections that used a hosted function are configured to use the same open model through
sentence-transformers, wherever the function appears in the collection's configuration or
schema. The `chromadb` client loads it from the collection, so `query_texts=` and
`add(documents=...)` keep working with no code change. Without the flag, the migration warns
about every collection that uses a hosted function.

Qwen is configured in float32; in bfloat16 its vectors drift slightly. If your collection used a
Qwen *task* (such as `nl_to_code`), queries need an instruction prefix:
[`examples/local_qwen_embeddings.py`](https://github.com/TheJustas228/pgchroma/blob/main/examples/local_qwen_embeddings.py)
shows how.

## Check before December 22

Run a few of your own queries on Chroma Cloud and on pgchroma while Chroma Cloud still answers,
and compare the top results. The models are the same, but the runtime differs, so confirm it on
your data.
