import json
import uuid

import numpy as np
import pytest
from test_migrate import FlakySource, seed_source

from pgchroma.cli import main
from pgchroma.export import DirectorySource, export
from pgchroma.migrate import MigrationError, migrate, verify
from pgchroma.source import ChromaSource, SourceError
from pgchroma.store import Store

pytest.importorskip("chromadb")

SEEDED = {"articles": 700, "products": 260, "tiny-ip": 3}


@pytest.fixture
async def store(pool):
    s = Store(pool, pool.pgchroma_info)
    yield s
    await s.wait_for_indexing()


def read_jsonl(path):
    lines = path.read_text().splitlines()
    return json.loads(lines[0])["collection"], [json.loads(line) for line in lines[1:]]


async def test_export_writes_each_collection_as_jsonl(real_chroma, tmp_path):
    from harness import make_client

    seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        report = await export(src, tmp_path, batch_size=100)
    assert {c.name: c.records for c in report.collections} == SEEDED

    header, records = read_jsonl(tmp_path / "default_database" / "articles.jsonl")
    assert header["name"] == "articles"
    assert header["metadata"] == {"owner": "team-a", "v": 2}
    assert header["configuration_json"]["hnsw"]["space"] == "cosine"
    assert len(records) == 700
    by_id = {r["id"]: r for r in records}
    assert by_id["articles-3"]["document"] == "articles document 3"
    assert by_id["articles-3"]["metadata"] == {"i": 3, "f": 1.0, "even": False, "tags": ["z"]}
    assert by_id["articles-3"]["uri"] == "s3://bucket/articles/3"

    # Every field matches what the stock client reads from Chroma.
    ids = ["articles-0", "articles-3", "articles-11"]
    want = (
        make_client(real_chroma.url)
        .get_collection("articles")
        .get(ids=ids, include=["embeddings", "documents", "metadatas", "uris"])
    )
    for i, rid in enumerate(want["ids"]):
        assert by_id[rid]["document"] == want["documents"][i]
        assert by_id[rid]["metadata"] == want["metadatas"][i]
        assert by_id[rid]["uri"] == want["uris"][i]
        assert np.array_equal(
            np.asarray(by_id[rid]["embedding"], dtype=np.float32),
            np.asarray(want["embeddings"][i], dtype=np.float32),
        )


async def test_migrate_from_an_export_matches_the_original(real_chroma, tmp_path, store):
    seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        await export(src, tmp_path, batch_size=100)
    async with DirectorySource(tmp_path) as backup:
        report = await migrate(backup, store, batch_size=100)
        assert {c.name: c.copied for c in report.collections} == SEEDED
        check = await verify(backup, store, sample=40, k=5)
    assert check.ok, check.problems
    assert all(c.query_overlap == 1.0 for c in check.collections)
    # The Postgres copy made from files is the same as the live Chroma data.
    async with ChromaSource(real_chroma.url) as src:
        check = await verify(src, store, sample=40, k=5)
    assert check.ok, check.problems


async def test_interrupted_export_resumes_without_duplicates(real_chroma, tmp_path):
    seed_source(real_chroma.url)
    async with FlakySource(real_chroma.url, fail_after=5) as src:
        with pytest.raises(ConnectionError):
            await export(src, tmp_path, batch_size=100, concurrency=1)
    with pytest.raises(SourceError, match="incomplete"):
        async with DirectorySource(tmp_path) as backup:
            await backup.databases()

    # A crash mid-write can also leave half a line at the end of the file.
    unfinished = max(tmp_path.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    with unfinished.open("a") as f:
        f.write('{"id": "half-writ')

    async with ChromaSource(real_chroma.url) as src:
        report = await export(src, tmp_path, batch_size=100)
    assert {c.name: c.records for c in report.collections} == SEEDED
    assert any(c.resumed for c in report.collections)
    for name, n in SEEDED.items():
        _, records = read_jsonl(tmp_path / "default_database" / f"{name}.jsonl")
        assert len({r["id"] for r in records}) == len(records) == n


def test_export_and_import_from_the_command_line(real_chroma, database_url, tmp_path, capsys):
    from harness import make_client

    c = make_client(real_chroma.url).create_collection(
        "notes", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    c.add(
        ids=[str(i) for i in range(50)],
        embeddings=[[float(i), 1.0, 2.0] for i in range(50)],
        documents=[f"note {i}" for i in range(50)],
    )
    out = tmp_path / "backup"
    assert main(["export", "--from", real_chroma.url, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "notes" in printed and "50" in printed
    assert len((out / "default_database" / "notes.jsonl").read_text().splitlines()) == 51

    schema = f"cli_{uuid.uuid4().hex[:8]}"
    code = main(["migrate", "--from", str(out), "--database-url", database_url, "--schema", schema])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "Verified" in printed


async def test_export_refuses_a_directory_holding_another_export(real_chroma, tmp_path):
    seed_source(real_chroma.url)
    async with ChromaSource(real_chroma.url) as src:
        await export(src, tmp_path, collections=["tiny-ip"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source"] = "https://api.trychroma.com"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    async with ChromaSource(real_chroma.url) as src:
        with pytest.raises(MigrationError, match="another --out"):
            await export(src, tmp_path)


async def test_export_and_migrate_an_empty_source(real_chroma, tmp_path, store):
    async with ChromaSource(real_chroma.url) as src:
        assert (await export(src, tmp_path)).collections == []
        assert (await migrate(src, store)).collections == []


async def test_export_from_a_pgchroma_server(pgchroma_server, tmp_path):
    from harness import make_client

    client = make_client(pgchroma_server.url)
    for c in client.list_collections():
        client.delete_collection(c.name)
    c = client.create_collection("kept", embedding_function=None)
    c.add(ids=["a", "b"], embeddings=[[1.0, 0.0], [0.0, 1.0]], metadatas=[{"k": 1}, None])
    async with ChromaSource(pgchroma_server.url) as src:
        report = await export(src, tmp_path)
    assert [(e.name, e.records) for e in report.collections] == [("kept", 2)]
    _, records = read_jsonl(tmp_path / "default_database" / "kept.jsonl")
    assert records == [
        {"id": "a", "embedding": [1.0, 0.0], "document": None, "metadata": {"k": 1}, "uri": None},
        {"id": "b", "embedding": [0.0, 1.0], "document": None, "metadata": None, "uri": None},
    ]
