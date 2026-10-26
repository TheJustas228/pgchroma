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
        self.log_path = os.path.join(self.dir, "chroma.log")
        self.log = open(self.log_path, "wb")  # noqa: SIM115 - closed in stop()
        # Bind to 127.0.0.1 explicitly: "localhost" may resolve to ::1 (e.g. on CI runners).
        self.proc = subprocess.Popen(
            [binary, "run", "--path", self.dir, "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        )
        self.url = f"http://127.0.0.1:{self.port}"
        try:
            wait_for(self.url, timeout=20)
        except RuntimeError as e:
            self.proc.kill()
            self.log.close()
            with open(self.log_path, errors="replace") as f:
                tail = f.read()[-3000:]
            raise RuntimeError(f"{e}; exit code {self.proc.poll()}; output:\n{tail}") from None

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.log.close()
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


def normalize(value: Any) -> Any:
    """Make client results comparable: numpy -> plain Python containers and numbers."""
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k != "data"}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def assert_same(got: Any, expected: Any, path: str = "$", rel: float = 1e-5) -> None:
    """Deep equality where floats may differ by float32 rounding (Chroma computes in f32)."""
    if isinstance(expected, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        assert type(got) is float, f"{path}: expected a float, got {got!r}"
        assert abs(got - expected) <= rel * max(1.0, abs(expected)), (
            f"{path}: {got!r} != {expected!r}"
        )
        return
    assert type(got) is type(expected), (
        f"{path}: type {type(got).__name__} != {type(expected).__name__} ({got!r} vs {expected!r})"
    )
    if isinstance(expected, dict):
        assert set(got) == set(expected), f"{path}: keys {sorted(got)} != {sorted(expected)}"
        for k in expected:
            assert_same(got[k], expected[k], f"{path}.{k}", rel)
    elif isinstance(expected, (list, tuple)):
        assert len(got) == len(expected), (
            f"{path}: length {len(got)} != {len(expected)}: {got!r} vs {expected!r}"
        )
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            assert_same(g, e, f"{path}[{i}]", rel)
    else:
        assert got == expected, f"{path}: {got!r} != {expected!r}"


def outcome(fn, client) -> tuple[str, Any]:
    """Run `fn(client)` and describe what happened in a comparable way."""
    try:
        return ("ok", normalize(fn(client)))
    except Exception as e:  # noqa: BLE001 - we compare whatever the client raises
        return ("error", type(e).__name__)
