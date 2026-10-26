"""pgchroma quickstart: ordinary Chroma client code; the data lands in Postgres.

pgchroma serve --database-url postgresql://...      # or: docker compose up
python examples/quickstart.py
"""

import os

import chromadb

client = chromadb.HttpClient(host="localhost", port=int(os.environ.get("PGCHROMA_PORT", "8000")))

notes = client.get_or_create_collection(
    "quickstart-notes", configuration={"hnsw": {"space": "cosine"}}
)
notes.upsert(
    ids=["pg", "cats", "rust"],
    documents=[
        "pgchroma keeps Chroma collections in Postgres tables",
        "Cats sleep for most of the day",
        "Rust's borrow checker prevents data races",
    ],
    metadatas=[{"topic": "databases"}, {"topic": "animals"}, {"topic": "programming"}],
)

hits = notes.query(
    query_texts=["where are my embeddings stored?"], n_results=2, include=["documents", "distances"]
)
for doc, dist in zip(hits["documents"][0], hits["distances"][0]):
    print(f"{dist:.3f}  {doc}")

print(
    notes.get(where={"topic": {"$in": ["animals", "programming"]}}, include=["metadatas"])[
        "metadatas"
    ]
)
print(notes.count(), "records in", client.get_collection("quickstart-notes").name)
