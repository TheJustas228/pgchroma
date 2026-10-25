"""Run the real Chroma server and pgchroma side by side and compare what the client sees."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any

import httpx
import numpy as np


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url + "/api/v2/heartbeat", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not start")


class ChromaProcess:
    """The real Chroma server from the `chromadb` wheel (`chroma run`)."""

    def __init__(self):
        binary = shutil.which("chroma")
        if binary is None:
            raise FileNotFoundError("chroma binary not found")
        self.port = free_port()
        self.dir = tempfile.mkdtemp(prefix="chroma-real-")
        self.proc = subprocess.Popen(
            [binary, "run", "--path", self.dir, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        )
        self.url = f"http://127.0.0.1:{self.port}"
        wait_for(self.url)

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)
        shutil.rmtree(self.dir, ignore_errors=True)


class PgChromaThread:
    """pgchroma served by uvicorn in a background thread, in its own schema."""

    def __init__(self, database_url: str, **settings: Any):
        import uvicorn

        from pgchroma.server import Settings, create_app

        self.schema = settings.pop("schema", f"compat_{uuid.uuid4().hex[:10]}")
        self.port = free_port()
        app = create_app(Settings(database_url=database_url, schema=self.schema, **settings))
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.port}"
        wait_for(self.url)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def make_client(url: str):
    import chromadb
    from chromadb.config import Settings

    host, port = url.removeprefix("http://").split(":")
    return chromadb.HttpClient(
        host=host, port=int(port), settings=Settings(anonymized_telemetry=False)
    )


def normalize(value: Any, digits: int = 5) -> Any:
    """Make client results comparable: numpy -> lists, floats rounded."""
    if isinstance(value, np.ndarray):
        return normalize(value.tolist(), digits)
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: normalize(v, digits) for k, v in value.items() if k != "data"}
    if isinstance(value, (list, tuple)):
        return [normalize(v, digits) for v in value]
    return value


def outcome(fn, client) -> tuple[str, Any]:
    """Run `fn(client)` and describe what happened in a comparable way."""
    try:
        return ("ok", normalize(fn(client)))
    except Exception as e:  # noqa: BLE001 - we compare whatever the client raises
        return ("error", type(e).__name__)
