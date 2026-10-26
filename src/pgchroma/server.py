"""Chroma v2 HTTP API on top of `Store`."""

from __future__ import annotations

import contextlib
import hmac
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjson
from psycopg import sql
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from . import __version__
from . import search as search_api
from .db import bootstrap, open_pool
from .errors import (
    AuthError,
    ChromaError,
    InternalError,
    InvalidArgumentError,
    NotFoundError,
    NotSupportedError,
)
from .store import Store
from .vectors import decode_embeddings

log = logging.getLogger("pgchroma")

API_VERSION = "1.0.0"
DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"
_PUBLIC_PATHS = {"/api/v2/heartbeat", "/api/v2/version", "/api/v2/healthcheck"}


@dataclass
class Settings:
    database_url: str
    schema: str = "pgchroma"
    auth_tokens: list[str] = field(default_factory=list)
    max_batch_size: int = 5461
    allow_reset: bool = False
    index_threshold: int = 10_000
    pool_min_size: int = 1
    pool_max_size: int = 10


class JSON(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


def _error_response(err: ChromaError) -> JSON:
    return JSON(err.to_dict(), status_code=err.status_code)


async def _body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise InvalidArgumentError(f"Invalid JSON body: {e}") from e
    if not isinstance(data, dict):
        raise InvalidArgumentError("Expected the request body to be a JSON object")
    return data


def _int_param(request: Request, name: str) -> int | None:
    value = request.query_params.get(name)
    if value is None or value == "":
        return None
    try:
        out = int(value)
    except ValueError as e:
        raise InvalidArgumentError(f"Expected {name} to be an integer, got {value!r}") from e
    if out < 0:
        raise InvalidArgumentError(f"Expected {name} to be non-negative, got {out}")
    return out


def _f32(rows: list | None) -> Any:
    """Distances and embeddings go out as float32, like Chroma's server."""
    if rows is None:
        return None
    return [np.asarray(r, dtype=np.float32) for r in rows]


class API:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store: Store | None = None

    # ------------------------------------------------------------ plumbing

    @property
    def s(self) -> Store:
        assert self.store is not None
        return self.store

    def _path(self, request: Request) -> tuple[str, str]:
        p = request.path_params
        return p["tenant"], p["database"]

    async def _coll(self, request: Request):
        tenant, database = self._path(request)
        return await self.s.collection(tenant, database, request.path_params["collection_id"])

    # --------------------------------------------------------------- system

    async def heartbeat(self, request: Request) -> JSON:
        return JSON({"nanosecond heartbeat": time.time_ns()})

    async def version(self, request: Request) -> JSON:
        return JSON(API_VERSION)

    async def healthcheck(self, request: Request) -> JSON:
        async with self.s.pool.connection() as conn:
            await conn.execute("SELECT 1")
        return JSON({"is_executor_ready": True, "is_log_client_ready": True})

    async def pre_flight_checks(self, request: Request) -> JSON:
        return JSON(
            {"max_batch_size": self.settings.max_batch_size, "supports_base64_encoding": True}
        )

    async def identity(self, request: Request) -> JSON:
        dbs = await self.s.list_databases(DEFAULT_TENANT, None, None)
        return JSON(
            {"user_id": "", "tenant": DEFAULT_TENANT, "databases": [d["name"] for d in dbs]}
        )

    async def reset(self, request: Request) -> JSON:
        if not self.settings.allow_reset:
            raise AuthError(
                "Resetting is not allowed by this configuration (start pgchroma with --allow-reset)"
            )
        schema = self.settings.schema
        async with self.s.pool.connection() as conn:
            await conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await bootstrap(self.s.pool, schema)
        self.s.clear_cache()
        return JSON(True)

    # ---------------------------------------------------- tenants/databases

    async def create_tenant(self, request: Request) -> JSON:
        body = await _body(request)
        await self.s.create_tenant(body.get("name"))
        return JSON({})

    async def get_tenant(self, request: Request) -> JSON:
        return JSON(await self.s.get_tenant(request.path_params["tenant"]))

    async def update_tenant(self, request: Request) -> JSON:
        body = await _body(request)
        await self.s.update_tenant(request.path_params["tenant"], body.get("resource_name"))
        return JSON({})

    async def list_databases(self, request: Request) -> JSON:
        return JSON(
            await self.s.list_databases(
                request.path_params["tenant"],
                _int_param(request, "limit"),
                _int_param(request, "offset"),
            )
        )

    async def create_database(self, request: Request) -> JSON:
        body = await _body(request)
        await self.s.create_database(request.path_params["tenant"], body.get("name"))
        return JSON({})

    async def get_database(self, request: Request) -> JSON:
        return JSON(await self.s.get_database(*self._path(request)))

    async def delete_database(self, request: Request) -> JSON:
        await self.s.delete_database(*self._path(request))
        return JSON({})

    # ----------------------------------------------------------- collections

    async def list_collections(self, request: Request) -> JSON:
        return JSON(
            await self.s.list_collections(
                *self._path(request), _int_param(request, "limit"), _int_param(request, "offset")
            )
        )

    async def count_collections(self, request: Request) -> JSON:
        return JSON(await self.s.count_collections(*self._path(request)))

    async def create_collection(self, request: Request) -> JSON:
        body = await _body(request)
        return JSON(
            await self.s.create_collection(
                *self._path(request),
                body.get("name"),
                body.get("metadata"),
                body.get("configuration"),
                body.get("schema"),
                bool(body.get("get_or_create", False)),
            )
        )

    async def get_collection(self, request: Request) -> JSON:
        tenant, database = self._path(request)
        name = request.path_params["collection_id"]
        try:
            return JSON(await self.s.get_collection(tenant, database, name))
        except NotFoundError as not_found:
            try:
                cid = uuid.UUID(name)
            except ValueError:
                raise not_found from None
            return JSON((await self.s.collection(tenant, database, cid)).to_json())

    async def get_collection_by_id(self, request: Request) -> JSON:
        return JSON((await self._coll(request)).to_json())

    async def get_collection_by_crn(self, request: Request) -> JSON:
        parts = request.path_params["crn"].split(":")
        if len(parts) != 3:
            raise InvalidArgumentError(
                "Expected a collection resource name of the form tenant:database:collection"
            )
        return JSON(await self.s.get_collection(*parts))

    async def update_collection(self, request: Request) -> JSON:
        body = await _body(request)
        await self.s.update_collection(
            *self._path(request),
            request.path_params["collection_id"],
            body.get("new_name"),
            body.get("new_metadata"),
            body.get("new_configuration"),
        )
        return JSON({})

    async def delete_collection(self, request: Request) -> JSON:
        tenant, database = self._path(request)
        name = request.path_params["collection_id"]
        try:
            await self.s.delete_collection(tenant, database, name)
        except NotFoundError as not_found:
            try:
                cid = uuid.UUID(name)
            except ValueError:
                raise not_found from None
            ref = await self.s.collection(tenant, database, cid)
            await self.s.delete_collection(tenant, database, ref.name)
        return JSON({})

    async def fork_collection(self, request: Request) -> JSON:
        body = await _body(request)
        return JSON(
            await self.s.fork_collection(
                *self._path(request), request.path_params["collection_id"], body.get("new_name")
            )
        )

    async def fork_count(self, request: Request) -> JSON:
        count = await self.s.fork_count(*self._path(request), request.path_params["collection_id"])
        return JSON({"count": count})

    async def indexing_status(self, request: Request) -> JSON:
        coll = await self._coll(request)
        total = await self.s.count(coll)
        progress = await self.s.indexing_progress(coll)
        indexed = int(total * progress)
        return JSON(
            {
                "num_indexed_ops": indexed,
                "num_unindexed_ops": total - indexed,
                "total_ops": total,
                "op_indexing_progress": progress,
            }
        )

    # --------------------------------------------------------------- records

    def _batch(self, body: dict) -> tuple:
        ids = body.get("ids")
        if isinstance(ids, list) and len(ids) > self.settings.max_batch_size:
            raise InvalidArgumentError(
                f"Cannot submit more than {self.settings.max_batch_size:,} records at once"
            )
        return (
            ids,
            decode_embeddings(body.get("embeddings")),
            body.get("metadatas"),
            body.get("documents"),
            body.get("uris"),
        )

    async def add(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        await self.s.add(coll, *self._batch(body))
        return JSON({}, status_code=201)

    async def upsert(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        await self.s.upsert(coll, *self._batch(body))
        return JSON({})

    async def update(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        await self.s.update(coll, *self._batch(body))
        return JSON({})

    async def delete_records(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        deleted = await self.s.delete(
            coll, body.get("ids"), body.get("where"), body.get("where_document"), body.get("limit")
        )
        return JSON({"deleted": deleted})

    async def count(self, request: Request) -> JSON:
        return JSON(await self.s.count(await self._coll(request)))

    async def get(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        result = await self.s.get(
            coll,
            body.get("ids"),
            body.get("where"),
            body.get("where_document"),
            body.get("limit"),
            body.get("offset"),
            body.get("include", ["metadatas", "documents"]),
        )
        result["embeddings"] = _f32(result["embeddings"])
        return JSON(result)

    async def query(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        result = await self.s.query(
            coll,
            decode_embeddings(body.get("query_embeddings")),
            body.get("n_results", 10),
            body.get("ids"),
            body.get("where"),
            body.get("where_document"),
            body.get("include", ["metadatas", "documents", "distances"]),
        )
        result["distances"] = _f32(result["distances"])
        if result["embeddings"] is not None:
            result["embeddings"] = [_f32(group) for group in result["embeddings"]]
        return JSON(result)

    async def search(self, request: Request) -> JSON:
        body = await _body(request)
        coll = await self._coll(request)
        return JSON(await search_api.search(self.s, coll, body))

    async def not_supported(self, request: Request) -> JSON:
        raise NotSupportedError(f"{request.url.path} is not supported by pgchroma")

    async def v1(self, request: Request) -> JSON:
        return JSON(
            {
                "error": "ChromaError",
                "message": "The v1 API is not supported by pgchroma. Upgrade your chromadb client to >= 1.0.",
            },
            status_code=410,
        )


Handler = Callable[[Request], Awaitable[Response]]


class _TokenAuth:
    """Accept `x-chroma-token: <t>` or `Authorization: Bearer <t>`, like Chroma's token auth."""

    def __init__(self, app, tokens: list[str]):
        self.app = app
        self.tokens = [t.encode() for t in tokens]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in _PUBLIC_PATHS:
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope["headers"]}
        token = headers.get(b"x-chroma-token")
        if token is None:
            auth = headers.get(b"authorization", b"")
            if auth[:7].lower() == b"bearer ":
                token = auth[7:].strip()
        if token is None or not any(hmac.compare_digest(token, t) for t in self.tokens):
            return await _error_response(AuthError("Unauthorized"))(scope, receive, send)
        return await self.app(scope, receive, send)


class _VersionHeader:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append(
                    (b"x-pgchroma-version", __version__.encode())
                )
            await send(message)

        return await self.app(scope, receive, send_with_header)


def _wrap(handler: Handler) -> Handler:
    async def endpoint(request: Request) -> Response:
        try:
            return await handler(request)
        except ChromaError as e:
            return _error_response(e)
        except Exception as e:  # pragma: no cover - last resort
            log.exception("unhandled error on %s %s", request.method, request.url.path)
            return _error_response(InternalError(f"{type(e).__name__}: {e}"))

    return endpoint


def create_app(settings: Settings) -> Starlette:
    api = API(settings)
    tokens = [t for t in settings.auth_tokens if t]

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        pool = await open_pool(
            settings.database_url,
            settings.schema,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
        )
        info = await bootstrap(pool, settings.schema)
        api.store = Store(pool, info, index_threshold=settings.index_threshold)
        app.state.store = api.store
        log.info(
            "pgchroma %s ready (schema %s, pgvector %s)",
            __version__,
            settings.schema,
            ".".join(map(str, info.pgvector_version)),
        )
        try:
            yield
        finally:
            await api.store.close()
            await pool.close()

    T = "/api/v2/tenants/{tenant}"
    D = T + "/databases/{database}"
    C = D + "/collections/{collection_id}"

    def R(path: str, handler: Handler, *methods: str) -> Route:
        return Route(path, _wrap(handler), methods=list(methods))

    routes = [
        R("/api/v2/heartbeat", api.heartbeat, "GET"),
        R("/api/v2/version", api.version, "GET"),
        R("/api/v2/healthcheck", api.healthcheck, "GET"),
        R("/api/v2/pre-flight-checks", api.pre_flight_checks, "GET"),
        R("/api/v2/auth/identity", api.identity, "GET"),
        R("/api/v2/reset", api.reset, "POST"),
        R("/api/v2/collections/{crn}", api.get_collection_by_crn, "GET"),
        R("/api/v2/tenants", api.create_tenant, "POST"),
        R(T, api.get_tenant, "GET"),
        R(T, api.update_tenant, "PATCH"),
        R(T + "/databases", api.list_databases, "GET"),
        R(T + "/databases", api.create_database, "POST"),
        R(D, api.get_database, "GET"),
        R(D, api.delete_database, "DELETE"),
        R(D + "/collections", api.list_collections, "GET"),
        R(D + "/collections", api.create_collection, "POST"),
        R(D + "/collections_count", api.count_collections, "GET"),
        R(D + "/collections/by-id/{collection_id}", api.get_collection_by_id, "GET"),
        R(C, api.get_collection, "GET"),
        R(C, api.update_collection, "PUT"),
        R(C, api.delete_collection, "DELETE"),
        R(C + "/add", api.add, "POST"),
        R(C + "/upsert", api.upsert, "POST"),
        R(C + "/update", api.update, "POST"),
        R(C + "/delete", api.delete_records, "POST"),
        R(C + "/get", api.get, "POST"),
        R(C + "/query", api.query, "POST"),
        R(C + "/count", api.count, "GET"),
        R(C + "/search", api.search, "POST"),
        R(C + "/fork", api.fork_collection, "POST"),
        R(C + "/fork_count", api.fork_count, "GET"),
        R(C + "/indexing_status", api.indexing_status, "GET"),
        R(C + "/functions/attach", api.not_supported, "POST"),
        R(C + "/functions/{name}", api.not_supported, "GET"),
        R(C + "/attached_functions/{name}/detach", api.not_supported, "POST"),
        R("/api/v1/{rest:path}", api.v1, "GET", "POST", "PUT", "PATCH", "DELETE"),
    ]
    middleware = [Middleware(_VersionHeader)]
    if tokens:
        middleware.append(Middleware(_TokenAuth, tokens=tokens))
    return Starlette(routes=routes, lifespan=lifespan, middleware=middleware)


def settings_from_env() -> Settings:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return Settings(
        database_url=url,
        schema=os.environ.get("PGCHROMA_SCHEMA", "pgchroma"),
        auth_tokens=[
            t.strip() for t in os.environ.get("PGCHROMA_AUTH_TOKENS", "").split(",") if t.strip()
        ],
        max_batch_size=int(os.environ.get("PGCHROMA_MAX_BATCH_SIZE", "5461")),
        allow_reset=os.environ.get("PGCHROMA_ALLOW_RESET", "").lower() in ("1", "true", "yes"),
        index_threshold=int(os.environ.get("PGCHROMA_INDEX_THRESHOLD", "10000")),
        pool_max_size=int(os.environ.get("PGCHROMA_POOL_SIZE", "10")),
    )


def app_from_env() -> Starlette:
    """uvicorn factory: `uvicorn pgchroma.server:app_from_env --factory`."""
    return create_app(settings_from_env())
