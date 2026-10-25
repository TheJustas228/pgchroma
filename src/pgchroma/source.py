"""Read-only client for a Chroma v2 server: Chroma Cloud or self-hosted.

Only GET-style calls are made (`get`, `count`, `query`, listing), so it keeps working after
Chroma Cloud turns read-only on 2026-11-22.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from . import __version__

CHROMA_CLOUD_URL = "https://api.trychroma.com"
INCLUDE_ALL = ["embeddings", "documents", "metadatas", "uris"]


class SourceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"source returned {status}: {message}")
        self.status = status
        self.message = message

    @property
    def is_size_limit(self) -> bool:
        text = self.message.lower()
        return self.status in (400, 413, 422, 429) and any(
            w in text for w in ("limit", "quota", "too large", "too many", "exceed")
        )


class ChromaSource:
    def __init__(
        self,
        url: str = CHROMA_CLOUD_URL,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 6,
        headers: dict[str, str] | None = None,
    ):
        self.url = url.rstrip("/")
        h = {"content-type": "application/json", "user-agent": f"pgchroma/{__version__}"}
        if api_key:
            h["x-chroma-token"] = api_key
        h.update(headers or {})
        self.client = httpx.AsyncClient(base_url=f"{self.url}/api/v2", headers=h, timeout=timeout)
        self.tenant = tenant
        self.database = database
        self.max_retries = max_retries

    async def __aenter__(self) -> ChromaSource:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.request(method, path, **kwargs)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    try:
                        body = resp.json()
                        message = body.get("message") or body.get("error") or resp.text
                    except ValueError:
                        message = resp.text
                    raise SourceError(resp.status_code, str(message))
            await asyncio.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 20)
        raise AssertionError("unreachable")

    def _coll(self, database: str, collection_id: str) -> str:
        return f"/tenants/{self.tenant}/databases/{database}/collections/{collection_id}"

    async def resolve_tenant(self) -> str:
        if self.tenant is None:
            identity = await self._request("GET", "/auth/identity")
            self.tenant = identity["tenant"]
        return self.tenant

    async def databases(self) -> list[str]:
        if self.database:
            return [self.database]
        tenant = await self.resolve_tenant()
        names: list[str] = []
        offset = 0
        while True:
            page = await self._request(
                "GET", f"/tenants/{tenant}/databases", params={"limit": 100, "offset": offset}
            )
            names.extend(d["name"] for d in page)
            if len(page) < 100:
                return names
            offset += len(page)

    async def collections(self, database: str) -> list[dict]:
        tenant = await self.resolve_tenant()
        out: list[dict] = []
        offset = 0
        while True:
            page = await self._request(
                "GET",
                f"/tenants/{tenant}/databases/{database}/collections",
                params={"limit": 100, "offset": offset},
            )
            out.extend(page)
            if len(page) < 100:
                return out
            offset += len(page)

    async def count(self, database: str, collection_id: str) -> int:
        await self.resolve_tenant()
        return int(await self._request("GET", self._coll(database, collection_id) + "/count"))

    async def get_page(self, database: str, collection_id: str, offset: int, limit: int) -> dict:
        await self.resolve_tenant()
        return await self._request(
            "POST",
            self._coll(database, collection_id) + "/get",
            json={"limit": limit, "offset": offset, "include": INCLUDE_ALL},
        )

    async def get_ids(self, database: str, collection_id: str, ids: list[str]) -> dict:
        await self.resolve_tenant()
        return await self._request(
            "POST",
            self._coll(database, collection_id) + "/get",
            json={"ids": ids, "include": INCLUDE_ALL},
        )

    async def query(
        self, database: str, collection_id: str, embeddings: list, n_results: int
    ) -> dict:
        await self.resolve_tenant()
        return await self._request(
            "POST",
            self._coll(database, collection_id) + "/query",
            json={"query_embeddings": embeddings, "n_results": n_results, "include": ["distances"]},
        )
