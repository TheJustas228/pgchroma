import os
import subprocess
import sys
import uuid

import pytest
from harness import free_port, wait_for

from pgchroma import __version__
from pgchroma.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_doctor(database_url, capsys):
    schema = f"cli_{uuid.uuid4().hex[:8]}"
    assert main(["doctor", "--database-url", database_url, "--schema", schema]) == 0
    out = capsys.readouterr().out
    assert "pgvector" in out and "PostgreSQL" in out


def test_migrate_end_to_end(real_chroma, database_url, capsys):
    pytest.importorskip("chromadb")
    from harness import make_client

    cl = make_client(real_chroma.url)
    c = cl.create_collection(
        "notes", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    c.add(
        ids=[str(i) for i in range(50)],
        embeddings=[[float(i), 1.0, 2.0] for i in range(50)],
        documents=[f"note {i}" for i in range(50)],
    )
    schema = f"cli_{uuid.uuid4().hex[:8]}"
    code = main(
        ["migrate", "--from", real_chroma.url, "--database-url", database_url, "--schema", schema]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "notes" in out and "50" in out and "Verified" in out


def test_migrate_reports_missing_collection(real_chroma, database_url, capsys):
    schema = f"cli_{uuid.uuid4().hex[:8]}"
    code = main(
        [
            "migrate",
            "--from",
            real_chroma.url,
            "--collection",
            "nope",
            "--database-url",
            database_url,
            "--schema",
            schema,
        ]
    )
    assert code == 1
    assert "nope" in capsys.readouterr().out


def test_serve_subprocess(database_url):
    pytest.importorskip("chromadb")
    from harness import make_client

    port = free_port()
    schema = f"cli_{uuid.uuid4().hex[:8]}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pgchroma", "serve", "--port", str(port), "--schema", schema],
        env={**os.environ, "DATABASE_URL": database_url},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(f"http://127.0.0.1:{port}")
        client = make_client(f"http://127.0.0.1:{port}")
        col = client.create_collection("served", embedding_function=None)
        col.add(ids=["1"], embeddings=[[1.0, 2.0]])
        assert col.count() == 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)
