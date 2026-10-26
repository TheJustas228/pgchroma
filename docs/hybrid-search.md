---
title: Chroma hybrid search (BM25, SPLADE) on Postgres
description: Run Chroma's Search API with sparse vectors, BM25 weighting and Rrf hybrid search on Postgres + pgvector using pgchroma. Works with ChromaBm25EmbeddingFunction unchanged.
---

# Hybrid search on Postgres

Chroma Cloud added sparse vectors and the Search API for keyword and hybrid search. Single-node
Chroma rejects both ("Sparse vector is not supported for local chroma"). pgchroma runs them on
Postgres, scored the way Chroma's source code scores them.

## BM25 with Chroma's own embedding function

```python
# pip install chromadb snowballstemmer
import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

client = chromadb.HttpClient(host="localhost", port=8000)  # pgchroma
schema = Schema().create_index(
    config=SparseVectorIndexConfig(
        source_key=K.DOCUMENT, embedding_function=ChromaBm25EmbeddingFunction(), bm25=True
    ),
    key="bm25",
)
tickets = client.create_collection("tickets", schema=schema)
tickets.add(ids=["1", "2"], documents=["my refund never arrived", "the tracking link is broken"])

# keyword search
tickets.search(Search().rank(Knn(query="refund", key="bm25", limit=10)))

# hybrid: dense + BM25 fused with reciprocal rank fusion
tickets.search(
    Search()
    .rank(
        Rrf(
            [
                Knn(query="where is my money?", return_rank=True, limit=100, default=1000),
                Knn(
                    query="where is my money?",
                    key="bm25",
                    return_rank=True,
                    limit=100,
                    default=1000,
                ),
            ]
        )
    )
    .limit(10)
    .select(K.DOCUMENT, K.SCORE)
)
```

The client computes the sparse vectors (on `add` from the document, on search from the query
text); pgchroma stores them in the record's metadata and ranks them.

## How scoring works

Matching [Chroma's implementation](https://github.com/chroma-core/chroma/tree/main/rust/worker/src/execution/operators):

- A sparse `Knn` scores a record as `1 - query · document` (lower is better). Only records that
  share at least one dimension with the query are candidates.
- With `bm25=True` in the collection schema, each query weight is multiplied by
  `ln(1 + (N - n + 0.5) / (n + 0.5))`, where `N` is the number of records in the collection and
  `n` the number whose vector contains that dimension.
- `Rrf` is computed from each `Knn`'s rank: `-Σ weight / (k + rank)`. Records missing from a
  `Knn` get its `default`, or are dropped when `default` is not set.
- `group_by` with `MinK` / `MaxK` keeps the top k records per group, then re-sorts by score.

## Performance

Candidates come from Postgres's jsonb (GIN) index on the metadata; every candidate is then scored
exactly. On 12,000 support tickets with BM25 vectors a keyword search takes about 60 ms. There is
no inverted index for sparse vectors yet, so very large sparse collections are slower than on
Chroma Cloud.

## SPLADE

Collections that used Chroma Cloud's hosted `chroma-cloud-splade` can switch to the same open
model running locally; see [hosted embedding functions](embedding-functions.md).
