"""The official JavaScript client (chromadb on npm) must see the same results on both servers."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).parent / "js"


@pytest.fixture(scope="module")
def node_client():
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node/npm not installed")
    if not (JS / "node_modules" / "chromadb").exists():
        subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund", "--silent"],
            cwd=JS,
            check=True,
            timeout=300,
        )
    return JS


def run(node_client, port: int) -> dict:
    p = subprocess.run(
        ["node", "compat.mjs", "127.0.0.1", str(port)],
        cwd=node_client,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def test_js_client_matches_real_chroma(node_client, real_chroma, pgchroma_server):
    assert run(node_client, pgchroma_server.port) == run(node_client, real_chroma.port)
