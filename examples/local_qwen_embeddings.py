"""Replace Chroma Cloud's hosted `chroma-cloud-qwen` embedding function with the same model, locally.

Chroma Cloud's `chroma-cloud-qwen` runs the open model Qwen/Qwen3-Embedding-0.6B (1024 dims; see
`ChromaCloudQwenEmbeddingFunction` in the chromadb client). The embeddings you migrated stay as
they are; you only need the model for *new* documents and for query text.

    pip install chromadb sentence-transformers

Two cases:

1. Your collection used `chroma-cloud-qwen` without a task (the default). Migrate with
   `pgchroma migrate --local-embedding-functions ...` and nothing else changes: the client loads
   the local model from the collection's configuration, so `query_texts=` and `add(documents=)`
   keep working.

2. It used a task (for example `task="nl_to_code"`): Chroma Cloud then prefixes *queries* with an
   instruction. Use `QwenEmbeddings(task=...)` below to embed queries the same way and pass
   `query_embeddings=`.

Before Chroma Cloud stops serving reads (2026-12-22), compare a few of your own queries on Chroma
Cloud and on pgchroma to confirm the results match for your data.
"""

from __future__ import annotations

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

MODEL = "Qwen/Qwen3-Embedding-0.6B"

# The instruction Chroma's client sends for the nl_to_code task (from Qwen3-Embedding's task prompts).
TASK_INSTRUCTIONS = {
    "nl_to_code": "Given a question about coding, retrieval code or passage that can solve user's question",
}


class QwenEmbeddings(EmbeddingFunction[Documents]):
    """Qwen3-Embedding-0.6B in float32, normalized; queries get Qwen's instruction prefix for a task."""

    def __init__(self, task: str | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL, device=device, model_kwargs={"torch_dtype": "float32"}
        )
        self.instruction = TASK_INSTRUCTIONS.get(task) if task else None

    def __call__(self, input: Documents) -> Embeddings:  # documents: no instruction
        return list(self.model.encode(list(input), normalize_embeddings=True))

    def embed_query(self, input: Documents) -> Embeddings:
        prompt = f"Instruct: {self.instruction}\nQuery:" if self.instruction else None
        return list(self.model.encode(list(input), prompt=prompt, normalize_embeddings=True))


if __name__ == "__main__":
    client = chromadb.HttpClient(host="localhost", port=8000)
    qwen = QwenEmbeddings(task=None)
    docs = client.get_or_create_collection("qwen-demo", embedding_function=None)
    texts = ["Postgres stores the vectors.", "The cat sleeps on the keyboard."]
    docs.upsert(ids=["db", "cat"], documents=texts, embeddings=qwen(texts))
    hits = docs.query(query_embeddings=qwen.embed_query(["where is my data kept?"]), n_results=1)
    print(hits["documents"][0][0])
