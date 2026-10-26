"""LangChain on pgchroma: the stock `langchain-chroma` vector store, pointed at pgchroma.

    pip install langchain-chroma langchain-openai
    pgchroma serve --database-url postgresql://...      # in another terminal
    OPENAI_API_KEY=... python examples/langchain_rag.py

Set PGCHROMA_EXAMPLE_FAKE_EMBEDDINGS=1 to run without an API key (deterministic fake vectors).
"""

import os

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

if os.environ.get("PGCHROMA_EXAMPLE_FAKE_EMBEDDINGS"):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    embeddings = DeterministicFakeEmbedding(size=256)
else:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = chromadb.HttpClient(
    host=os.environ.get("PGCHROMA_HOST", "localhost"),
    port=int(os.environ.get("PGCHROMA_PORT", "8000")),
)
store = Chroma(client=client, collection_name="langchain-demo", embedding_function=embeddings)

store.add_documents(
    [
        Document(
            page_content="Refunds are processed within 5 business days.",
            metadata={"topic": "billing"},
        ),
        Document(
            page_content="You can export your data from Settings > Export.",
            metadata={"topic": "account"},
        ),
        Document(
            page_content="Invoices are emailed on the first day of the month.",
            metadata={"topic": "billing"},
        ),
    ],
    ids=["refunds", "export", "invoices"],
)

for doc, score in store.similarity_search_with_score(
    "when do I get my money back?", k=2, filter={"topic": "billing"}
):
    print(f"{score:.3f}  {doc.metadata['topic']:8}  {doc.page_content}")

retriever = store.as_retriever(search_kwargs={"k": 1})
print("retriever ->", retriever.invoke("how do I download my data?")[0].page_content)
