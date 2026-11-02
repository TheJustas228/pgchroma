"""LlamaIndex on pgchroma: use LLamaIndex's ChromaVectorStore pointed at pgchroma with local embeddings.

    pip install llama-index-core llama-index-vector-stores-chroma llama-index-embeddings-huggingface chromadb
    docker compose up
    python examples/llamaindex_rag.py

Uses a local HuggingFace embedding model (BAAI/bge-small-en-v1.5) so it runs without any API keys
"""

import os
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

if os.environ.get("PGCHROMA_EXAMPLE_FAKE_EMBEDDINGS"):
    from llama_index.core.embeddings import MockEmbedding
    embed_model = MockEmbedding(embed_dim=384)
else:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

client = chromadb.HttpClient(
    host=os.environ.get("PGCHROMA_HOST", "localhost"),
    port=int(os.environ.get("PGCHROMA_PORT", "8000")),
)

collection = client.get_or_create_collection("llamaindex-demo")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

nodes = [
    TextNode(
        text="Refunds are processed within 5 business days.",
        metadata={"topic": "billing"},
        id_="refunds",
    ),
    TextNode(
        text="You can export your data from Settings > Export.",
        metadata={"topic": "account"},
        id_="export",
    ),
    TextNode(
        text="Invoices are emailed on the first day of the month",
        metadata={"topic": "billing"},
        id_="invoices"
    ),
]

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

filters = MetadataFilters(
    filters=[ExactMatchFilter(key="topic", value="billing")]
)
retriever = index.as_retriever(similarity_top_k=2, filters=filters)

for node in retriever.retrieve("when do I get my money back?"):
    print(f"{node.score:.3f}  {node.metadata['topic']:8}  {node.text}")

general_retriever = index.as_retriever(similarity_top_k=1)
print("retriever ->", general_retriever.retrieve("how do I download my data?")[0].text)