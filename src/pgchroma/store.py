"""Postgres storage for tenants, databases, collections and records.

Every collection owns a table `c_<uuid hex>`:

    seq        bigint identity   -- insertion order (Chroma's `get` order)
    id         text primary key
    embedding  vector(dim)       -- typed on the first write that fixes the dimension
    document   text
    metadata   jsonb
    uri        text
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson
import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from .config import (
    default_schema,
    merge_configuration,
    normalize_configuration,
    validate_collection_name,
)
from .db import PgInfo
from .errors import ChromaError, DuplicateError, InternalError, InvalidArgumentError, NotFoundError
from .filters import compile_where, compile_where_document
from .indexing import ensure_index, index_plan

log = logging.getLogger("pgchroma")

INCLUDABLE = ("documents", "embeddings", "metadatas", "distances", "uris")
_RESERVED_METADATA_KEY = "chroma:document"

# ORDER BY operator and distance expression per space. Chroma reports squared L2,
# 1 - cosine similarity, and 1 - inner product.
_SPACE_SQL = {
    "l2": ("<->", "power(embedding <-> {q}, 2)"),
    "cosine": ("<=>", "(embedding <=> {q})"),
    "ip": ("<#>", "(1 + (embedding <#> {q}))"),
}


@dataclass
class CollectionRef:
    id: uuid.UUID
    name: str
    tenant: str
    database: str
    database_id: uuid.UUID
    metadata: dict | None
    configuration: dict
    schema: dict | None
    dimension: int | None
    space: str
    version: int
    log_position: int

    @property
    def table(self) -> sql.Identifier:
        return table_ident(self.id)

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "configuration_json": self.configuration,
            "schema": self.schema,
            "metadata": self.metadata,
            "dimension": self.dimension,
            "tenant": self.tenant,
            "database": self.database,
            "log_position": self.log_position,
            "version": self.version,
        }


def table_ident(collection_id: uuid.UUID) -> sql.Identifier:
    return sql.Identifier(f"c_{collection_id.hex}")


_COLLECTION_COLUMNS = (
    "c.id, c.name, d.tenant, d.name, c.database_id, c.metadata, c.configuration, c.schema, "
    "c.dimension, c.space, c.version, c.log_position"
)


def _ref(row) -> CollectionRef:
    return CollectionRef(*row)


def _parse_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except ValueError as e:
        raise InvalidArgumentError(f"Invalid collection id {value!r}") from e


@contextlib.contextmanager
def _translate_errors():
    try:
        yield
    except ChromaError:
        raise
    except psycopg.errors.InvalidRegularExpression as e:
        raise InvalidArgumentError(f"Invalid regex: {e.diag.message_primary}") from e
    except psycopg.errors.UndefinedTable as e:
        raise NotFoundError("Collection does not exist") from e
    except psycopg.Error as e:
        raise InternalError(f"{type(e).__name__}: {e}") from e


# ---------------------------------------------------------------- validation


def _validate_metadata_value(key: str, value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise InvalidArgumentError(f"Metadata value for {key!r} must be finite")
        return
    if isinstance(value, list):
        if not value:
            raise InvalidArgumentError(f"Expected metadata list value for {key!r} to be non-empty")
        kinds = {
            bool
            if isinstance(v, bool)
            else str
            if isinstance(v, str)
            else (int, float)
            if isinstance(v, (int, float))
            else None
            for v in value
        }
        if None in kinds or len(kinds) != 1:
            raise InvalidArgumentError(
                f"Expected metadata list value for {key!r} to contain only str, int, float or bool of one type"
            )
        return
    if isinstance(value, dict) and value.get("#type") == "sparse_vector":
        return
    raise InvalidArgumentError(
        f"Expected metadata value to be a str, int, float, bool, SparseVector, list, or None, got {value!r}"
    )


def validate_metadata(metadata: Any, *, what: str = "metadata") -> dict | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise InvalidArgumentError(f"Expected {what} to be a dict or None, got {metadata!r}")
    if not metadata:
        raise InvalidArgumentError(f"Expected {what} to be a non-empty dict, got {metadata!r}")
    for key, value in metadata.items():
        if key == _RESERVED_METADATA_KEY:
            raise InvalidArgumentError(f"Expected {what} to not contain the reserved key {key}")
        _validate_metadata_value(key, value)
    return metadata


def _check_lengths(ids: list, **fields: list | None) -> None:
    if not isinstance(ids, list) or not ids:
        raise InvalidArgumentError("Expected IDs to be a non-empty list")
    for i in ids:
        if not isinstance(i, str) or not i:
            raise InvalidArgumentError(f"Expected ID to be a non-empty str, got {i!r}")
    for name, value in fields.items():
        if value is not None and len(value) != len(ids):
            raise InvalidArgumentError(
                f"Unequal lengths for fields: ids: {len(ids)}, {name}: {len(value)}"
            )
    if len(set(ids)) != len(ids):
        seen, dups = set(), []
        for i in ids:
            if i in seen and i not in dups:
                dups.append(i)
            seen.add(i)
        raise InvalidArgumentError(
            f"Expected IDs to be unique, found duplicates of: {', '.join(dups[:10])}"
        )


def _check_optional_strings(name: str, values: list | None) -> None:
    if values is None:
        return
    for v in values:
        if v is not None and not isinstance(v, str):
            raise InvalidArgumentError(f"Expected {name} to be str or None, got {v!r}")


def _check_include(include: list | None, allowed=INCLUDABLE) -> list:
    include = list(include) if include is not None else []
    for item in include:
        if item not in allowed:
            raise InvalidArgumentError(
                f"Expected include item to be one of {', '.join(allowed)}, got {item}"
            )
    return include


def _dump_metadatas(metadatas: list | None, n: int, *, strip_nulls: bool) -> list[str | None]:
    out: list[str | None] = []
    for m in metadatas if metadatas is not None else [None] * n:
        m = validate_metadata(m)
        if m is None:
            out.append(None)
        else:
            if strip_nulls:
                m = {k: v for k, v in m.items() if v is not None}
            out.append(orjson.dumps(m).decode())
    return out


def _as_array(v: Any) -> np.ndarray | None:
    if v is None or isinstance(v, np.ndarray):
        return v
    return v.to_numpy()


def _has_null_values(metadatas: list | None) -> bool:
    return bool(metadatas) and any(
        isinstance(m, dict) and any(v is None for v in m.values()) for m in metadatas
    )


class Store:
    def __init__(self, pool: AsyncConnectionPool, info: PgInfo, *, index_threshold: int = 10_000):
        self.pool = pool
        self.info = info
        self.index_threshold = index_threshold
        self._indexed: set[uuid.UUID] = set()
        self._index_tasks: dict[uuid.UUID, asyncio.Task] = {}

    @contextlib.asynccontextmanager
    async def _tx(self) -> AsyncIterator[psycopg.AsyncConnection]:
        with _translate_errors():
            async with self.pool.connection() as conn, conn.transaction():
                yield conn

    # ------------------------------------------------------------- tenants

    async def create_tenant(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise InvalidArgumentError("Expected tenant name to be a non-empty str")
        async with self._tx() as conn:
            cur = await conn.execute(
                "INSERT INTO tenants (name) VALUES (%s) ON CONFLICT DO NOTHING", (name,)
            )
            if cur.rowcount == 0:
                raise DuplicateError(f"Tenant [{name}] already exists")

    async def get_tenant(self, name: str) -> dict:
        async with self._tx() as conn:
            row = await (
                await conn.execute(
                    "SELECT name, resource_name FROM tenants WHERE name = %s", (name,)
                )
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Tenant [{name}] not found")
        return {"name": row[0], "resource_name": row[1]}

    async def update_tenant(self, name: str, resource_name: str | None) -> None:
        async with self._tx() as conn:
            cur = await conn.execute(
                "UPDATE tenants SET resource_name = %s WHERE name = %s", (resource_name, name)
            )
            if cur.rowcount == 0:
                raise NotFoundError(f"Tenant [{name}] not found")

    # ----------------------------------------------------------- databases

    async def _tenant_exists(self, conn, tenant: str) -> None:
        row = await (
            await conn.execute("SELECT 1 FROM tenants WHERE name = %s", (tenant,))
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Tenant [{tenant}] not found")

    async def _database_id(self, conn, tenant: str, database: str) -> uuid.UUID:
        row = await (
            await conn.execute(
                "SELECT id FROM databases WHERE tenant = %s AND name = %s", (tenant, database)
            )
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Database [{database}] not found for tenant [{tenant}]")
        return row[0]

    async def create_database(self, tenant: str, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise InvalidArgumentError("Expected database name to be a non-empty str")
        async with self._tx() as conn:
            await self._tenant_exists(conn, tenant)
            cur = await conn.execute(
                "INSERT INTO databases (id, tenant, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (uuid.uuid4(), tenant, name),
            )
            if cur.rowcount == 0:
                raise DuplicateError(f"Database [{name}] already exists")

    async def get_database(self, tenant: str, name: str) -> dict:
        async with self._tx() as conn:
            db_id = await self._database_id(conn, tenant, name)
        return {"id": str(db_id), "name": name, "tenant": tenant}

    async def list_databases(
        self, tenant: str, limit: int | None, offset: int | None
    ) -> list[dict]:
        async with self._tx() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, name, tenant FROM databases WHERE tenant = %s "
                    "ORDER BY created_at, name LIMIT %s OFFSET %s",
                    (tenant, limit, offset or 0),
                )
            ).fetchall()
        return [{"id": str(r[0]), "name": r[1], "tenant": r[2]} for r in rows]

    async def delete_database(self, tenant: str, name: str) -> None:
        async with self._tx() as conn:
            db_id = await self._database_id(conn, tenant, name)
            ids = await (
                await conn.execute("SELECT id FROM collections WHERE database_id = %s", (db_id,))
            ).fetchall()
            for (cid,) in ids:
                await conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table_ident(cid)))
            await conn.execute("DELETE FROM databases WHERE id = %s", (db_id,))

    # --------------------------------------------------------- collections

    async def _create_table(self, conn, collection_id: uuid.UUID) -> None:
        t = table_ident(collection_id)
        await conn.execute(
            sql.SQL(
                "CREATE TABLE {} (seq bigint GENERATED ALWAYS AS IDENTITY, id text PRIMARY KEY, "
                "embedding vector, document text, metadata jsonb, uri text)"
            ).format(t)
        )
        await conn.execute(sql.SQL("CREATE INDEX ON {} (seq)").format(t))
        await conn.execute(
            sql.SQL("CREATE INDEX ON {} USING gin (metadata jsonb_path_ops)").format(t)
        )

    async def create_collection(
        self,
        tenant: str,
        database: str,
        name: str,
        metadata: dict | None,
        configuration: dict | None,
        schema: dict | None,
        get_or_create: bool,
    ) -> dict:
        validate_collection_name(name)
        validate_metadata(metadata, what="collection metadata")
        cfg = normalize_configuration(configuration, metadata)
        if schema is not None and not isinstance(schema, dict):
            raise InvalidArgumentError("Expected schema to be an object")
        schema = schema or default_schema(cfg)
        async with self._tx() as conn:
            db_id = await self._database_id(conn, tenant, database)
            new_id = uuid.uuid4()
            cur = await conn.execute(
                "INSERT INTO collections (id, database_id, name, metadata, configuration, schema, space) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s) "
                "ON CONFLICT (database_id, name) DO NOTHING",
                (
                    new_id,
                    db_id,
                    name,
                    None if metadata is None else orjson.dumps(metadata).decode(),
                    orjson.dumps(cfg).decode(),
                    orjson.dumps(schema).decode(),
                    cfg["hnsw"]["space"],
                ),
            )
            if cur.rowcount == 0:
                if not get_or_create:
                    raise DuplicateError(f"Collection [{name}] already exists")
            else:
                await self._create_table(conn, new_id)
            ref = await self._find(conn, tenant, database, name=name)
        return ref.to_json()

    async def _find(
        self, conn, tenant: str, database: str, *, name=None, cid=None
    ) -> CollectionRef:
        cond, value = ("c.name = %s", name) if name is not None else ("c.id = %s", cid)
        row = await (
            await conn.execute(
                f"SELECT {_COLLECTION_COLUMNS} FROM collections c JOIN databases d ON d.id = c.database_id "
                f"WHERE d.tenant = %s AND d.name = %s AND {cond}",
                (tenant, database, value),
            )
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Collection [{name if name is not None else cid}] does not exist")
        return _ref(row)

    async def get_collection(self, tenant: str, database: str, name: str) -> dict:
        async with self._tx() as conn:
            return (await self._find(conn, tenant, database, name=name)).to_json()

    async def collection(self, tenant: str, database: str, collection_id: Any) -> CollectionRef:
        cid = _parse_uuid(collection_id)
        async with self._tx() as conn:
            return await self._find(conn, tenant, database, cid=cid)

    async def list_collections(
        self, tenant: str, database: str, limit: int | None, offset: int | None
    ) -> list[dict]:
        async with self._tx() as conn:
            await self._database_id(conn, tenant, database)
            rows = await (
                await conn.execute(
                    f"SELECT {_COLLECTION_COLUMNS} FROM collections c JOIN databases d ON d.id = c.database_id "
                    "WHERE d.tenant = %s AND d.name = %s ORDER BY c.id LIMIT %s OFFSET %s",
                    (tenant, database, limit, offset or 0),
                )
            ).fetchall()
        return [_ref(r).to_json() for r in rows]

    async def count_collections(self, tenant: str, database: str) -> int:
        async with self._tx() as conn:
            db_id = await self._database_id(conn, tenant, database)
            row = await (
                await conn.execute(
                    "SELECT count(*) FROM collections WHERE database_id = %s", (db_id,)
                )
            ).fetchone()
        return row[0]

    async def update_collection(
        self,
        tenant: str,
        database: str,
        collection_id: Any,
        new_name: str | None,
        new_metadata: dict | None,
        new_configuration: dict | None,
    ) -> None:
        cid = _parse_uuid(collection_id)
        if new_name is not None:
            validate_collection_name(new_name)
        validate_metadata(new_metadata, what="collection metadata")
        async with self._tx() as conn:
            ref = await self._find(conn, tenant, database, cid=cid)
            if new_name is not None and new_name != ref.name:
                taken = await (
                    await conn.execute(
                        "SELECT 1 FROM collections WHERE database_id = %s AND name = %s",
                        (ref.database_id, new_name),
                    )
                ).fetchone()
                if taken:
                    raise DuplicateError(f"Collection [{new_name}] already exists")
            cfg = merge_configuration(ref.configuration, new_configuration)
            schema = default_schema(cfg) if new_configuration else ref.schema
            await conn.execute(
                "UPDATE collections SET name = %s, metadata = %s::jsonb, configuration = %s::jsonb, "
                "schema = %s::jsonb, updated_at = now() WHERE id = %s",
                (
                    new_name if new_name is not None else ref.name,
                    orjson.dumps(new_metadata).decode()
                    if new_metadata is not None
                    else (None if ref.metadata is None else orjson.dumps(ref.metadata).decode()),
                    orjson.dumps(cfg).decode(),
                    None if schema is None else orjson.dumps(schema).decode(),
                    cid,
                ),
            )

    async def delete_collection(self, tenant: str, database: str, name: str) -> None:
        async with self._tx() as conn:
            ref = await self._find(conn, tenant, database, name=name)
            await conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(ref.table))
            await conn.execute("DELETE FROM collections WHERE id = %s", (ref.id,))

    async def fork_collection(
        self, tenant: str, database: str, collection_id: Any, new_name: str
    ) -> dict:
        validate_collection_name(new_name)
        cid = _parse_uuid(collection_id)
        async with self._tx() as conn:
            src = await self._find(conn, tenant, database, cid=cid)
            new_id = uuid.uuid4()
            cur = await conn.execute(
                "INSERT INTO collections (id, database_id, name, metadata, configuration, schema, "
                "dimension, space, forked_from) SELECT %s, database_id, %s, metadata, configuration, "
                "schema, dimension, space, id FROM collections WHERE id = %s "
                "ON CONFLICT (database_id, name) DO NOTHING",
                (new_id, new_name, cid),
            )
            if cur.rowcount == 0:
                raise DuplicateError(f"Collection [{new_name}] already exists")
            await conn.execute(
                sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING ALL)").format(
                    table_ident(new_id), src.table
                )
            )
            await conn.execute(
                sql.SQL(
                    "INSERT INTO {} (id, embedding, document, metadata, uri) "
                    "SELECT id, embedding, document, metadata, uri FROM {} ORDER BY seq"
                ).format(table_ident(new_id), src.table)
            )
            return (await self._find(conn, tenant, database, cid=new_id)).to_json()

    async def fork_count(self, tenant: str, database: str, collection_id: Any) -> int:
        cid = _parse_uuid(collection_id)
        async with self._tx() as conn:
            await self._find(conn, tenant, database, cid=cid)
            row = await (
                await conn.execute(
                    "SELECT count(*) FROM collections WHERE forked_from = %s", (cid,)
                )
            ).fetchone()
        return row[0]

    # ------------------------------------------------------------- records

    async def _fix_dimension(self, conn, coll: CollectionRef, dim: int) -> None:
        """Validate `dim`; the first write of a collection fixes its dimension and column type."""
        if coll.dimension is None:
            row = await (
                await conn.execute(
                    "SELECT dimension FROM collections WHERE id = %s FOR UPDATE", (coll.id,)
                )
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Collection [{coll.id}] does not exist")
            if row[0] is None:
                await conn.execute(
                    sql.SQL("ALTER TABLE {} ALTER COLUMN embedding TYPE vector({})").format(
                        coll.table, sql.Literal(dim)
                    )
                )
                await conn.execute(
                    "UPDATE collections SET dimension = %s WHERE id = %s", (dim, coll.id)
                )
            coll.dimension = row[0] if row[0] is not None else dim
        if coll.dimension != dim:
            raise InvalidArgumentError(
                f"Collection expecting embedding with dimension of {coll.dimension}, got {dim}"
            )

    def _prepare_write(self, ids, embeddings, metadatas, documents, uris, *, need_embeddings: bool):
        _check_lengths(
            ids,
            embeddings=None if embeddings is None else list(range(len(embeddings))),
            metadatas=metadatas,
            documents=documents,
            uris=uris,
        )
        if need_embeddings and embeddings is None:
            raise InvalidArgumentError("Expected embeddings to be provided")
        _check_optional_strings("document", documents)
        _check_optional_strings("uri", uris)

    async def add(self, coll, ids, embeddings, metadatas, documents, uris) -> None:
        self._prepare_write(ids, embeddings, metadatas, documents, uris, need_embeddings=True)
        metas = _dump_metadatas(metadatas, len(ids), strip_nulls=True)
        async with self._tx() as conn:
            await self._fix_dimension(conn, coll, embeddings.shape[1])
            await conn.execute(
                sql.SQL(
                    "INSERT INTO {} (id, embedding, document, metadata, uri) "
                    "SELECT u.id, u.embedding, u.document, u.metadata, u.uri "
                    "FROM unnest(%s::text[], %s::vector[], %s::text[], %s::jsonb[], %s::text[]) "
                    "WITH ORDINALITY AS u(id, embedding, document, metadata, uri, ord) "
                    "ORDER BY u.ord ON CONFLICT (id) DO NOTHING"
                ).format(coll.table),
                (
                    ids,
                    list(embeddings),
                    documents or [None] * len(ids),
                    metas,
                    uris or [None] * len(ids),
                ),
            )
        await self._after_write(coll)

    async def upsert(self, coll, ids, embeddings, metadatas, documents, uris) -> None:
        self._prepare_write(ids, embeddings, metadatas, documents, uris, need_embeddings=True)
        # Keep nulls: on conflict they delete keys; for fresh rows they are stripped below.
        metas = _dump_metadatas(metadatas, len(ids), strip_nulls=False)
        async with self._tx() as conn:
            await self._fix_dimension(conn, coll, embeddings.shape[1])
            cur = await conn.execute(
                sql.SQL(
                    "INSERT INTO {} AS r (id, embedding, document, metadata, uri) "
                    "SELECT u.id, u.embedding, u.document, u.metadata, u.uri "
                    "FROM unnest(%s::text[], %s::vector[], %s::text[], %s::jsonb[], %s::text[]) "
                    "WITH ORDINALITY AS u(id, embedding, document, metadata, uri, ord) ORDER BY u.ord "
                    "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                    "document = COALESCE(EXCLUDED.document, r.document), "
                    "uri = COALESCE(EXCLUDED.uri, r.uri), "
                    "metadata = CASE WHEN EXCLUDED.metadata IS NULL THEN r.metadata "
                    "ELSE jsonb_strip_nulls(COALESCE(r.metadata, '{{}}'::jsonb) || EXCLUDED.metadata) END "
                    "RETURNING (xmax = 0) AS inserted, id"
                ).format(coll.table),
                (
                    ids,
                    list(embeddings),
                    documents or [None] * len(ids),
                    metas,
                    uris or [None] * len(ids),
                ),
            )
            if _has_null_values(metadatas):
                fresh = [r[1] for r in await cur.fetchall() if r[0]]
                if fresh:
                    await conn.execute(
                        sql.SQL(
                            "UPDATE {} SET metadata = jsonb_strip_nulls(metadata) WHERE id = ANY(%s)"
                        ).format(coll.table),
                        (fresh,),
                    )
        await self._after_write(coll)

    async def update(self, coll, ids, embeddings, metadatas, documents, uris) -> None:
        self._prepare_write(ids, embeddings, metadatas, documents, uris, need_embeddings=False)
        metas = _dump_metadatas(metadatas, len(ids), strip_nulls=False)
        n = len(ids)
        async with self._tx() as conn:
            if embeddings is not None:
                await self._fix_dimension(conn, coll, embeddings.shape[1])
            await conn.execute(
                sql.SQL(
                    "UPDATE {} AS r SET embedding = COALESCE(s.embedding, r.embedding), "
                    "document = COALESCE(s.document, r.document), uri = COALESCE(s.uri, r.uri), "
                    "metadata = CASE WHEN s.metadata IS NULL THEN r.metadata "
                    "ELSE jsonb_strip_nulls(COALESCE(r.metadata, '{{}}'::jsonb) || s.metadata) END "
                    "FROM unnest(%s::text[], %s::vector[], %s::text[], %s::jsonb[], %s::text[]) "
                    "AS s(id, embedding, document, metadata, uri) WHERE r.id = s.id"
                ).format(coll.table),
                (
                    ids,
                    list(embeddings) if embeddings is not None else [None] * n,
                    documents or [None] * n,
                    metas,
                    uris or [None] * n,
                ),
            )

    async def _after_write(self, coll: CollectionRef) -> None:
        """Start a background HNSW build once the collection is big enough (see indexing.py)."""
        if coll.id in self._indexed or coll.id in self._index_tasks:
            return
        plan = index_plan(coll.dimension, coll.space, self.info)
        if plan is None:
            return
        table = f"c_{coll.id.hex}"

        async def build() -> None:
            try:
                if await ensure_index(
                    self.pool, table, plan, coll.configuration.get("hnsw", {}), self.index_threshold
                ):
                    self._indexed.add(coll.id)
            except Exception:  # never fail a write because an index build failed
                log.exception("HNSW index build failed for %s", table)
            finally:
                self._index_tasks.pop(coll.id, None)

        self._index_tasks[coll.id] = asyncio.create_task(build())

    async def wait_for_indexing(self) -> None:
        """Wait for background index builds (tests and graceful shutdown)."""
        while self._index_tasks:
            await asyncio.gather(*list(self._index_tasks.values()), return_exceptions=True)

    def _filters(self, ids, where, where_document) -> tuple[sql.Composable, list]:
        w_sql, w_params = compile_where(where)
        d_sql, d_params = compile_where_document(where_document)
        parts = [sql.SQL(w_sql), sql.SQL(d_sql)]
        params = w_params + d_params
        if ids is not None:
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise InvalidArgumentError("Expected ids to be a list of str")
            parts.append(sql.SQL("id = ANY(%s)"))
            params.append(ids)
        return sql.SQL(" AND ").join(sql.SQL("({})").format(p) for p in parts), params

    async def delete(self, coll, ids, where, where_document, limit) -> int:
        if ids is None and not where and not where_document:
            raise InvalidArgumentError(
                "You must provide either ids, where, or where_document to delete."
            )
        cond, params = self._filters(ids, where, where_document)
        async with self._tx() as conn:
            if limit is None:
                cur = await conn.execute(
                    sql.SQL("DELETE FROM {} WHERE {}").format(coll.table, cond), params
                )
            else:
                cur = await conn.execute(
                    sql.SQL(
                        "DELETE FROM {t} WHERE id IN (SELECT id FROM {t} WHERE {c} ORDER BY seq LIMIT %s)"
                    ).format(t=coll.table, c=cond),
                    [*params, int(limit)],
                )
            return cur.rowcount

    async def count(self, coll) -> int:
        async with self._tx() as conn:
            row = await (
                await conn.execute(sql.SQL("SELECT count(*) FROM {}").format(coll.table))
            ).fetchone()
        return row[0]

    @staticmethod
    def _select_columns(include: list, prefix: str = "") -> list[str]:
        cols = []
        for item, col in (
            ("embeddings", "embedding"),
            ("documents", "document"),
            ("metadatas", "metadata"),
            ("uris", "uri"),
        ):
            if item not in include:
                continue
            if col == "metadata" and not prefix:
                # Chroma keeps uris inside the metadata record, so a record with a uri and
                # no metadata reads back as {} rather than null.
                cols.append(
                    "CASE WHEN metadata IS NULL AND uri IS NOT NULL THEN '{}'::jsonb "
                    "ELSE metadata END AS metadata"
                )
            else:
                cols.append(prefix + col)
        return cols

    async def get(self, coll, ids, where, where_document, limit, offset, include) -> dict:
        include = _check_include(include)
        cond, params = self._filters(ids, where, where_document)
        cols = self._select_columns(include)
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise InvalidArgumentError(f"Expected limit to be a non-negative int, got {limit!r}")
        if offset is not None and (not isinstance(offset, int) or offset < 0):
            raise InvalidArgumentError(f"Expected offset to be a non-negative int, got {offset!r}")
        async with self._tx() as conn:
            rows = await (
                await conn.execute(
                    sql.SQL("SELECT {} FROM {} WHERE {} ORDER BY seq LIMIT %s OFFSET %s").format(
                        sql.SQL(", ").join(sql.SQL(c) for c in ["id", *cols]), coll.table, cond
                    ),
                    [*params, limit, offset or 0],
                )
            ).fetchall()
        result: dict[str, Any] = {"ids": [r[0] for r in rows]}
        pos = 1
        for item in ("embeddings", "documents", "metadatas", "uris"):
            if item in include:
                result[item] = [_as_array(r[pos]) if item == "embeddings" else r[pos] for r in rows]
                pos += 1
            else:
                result[item] = None
        result["include"] = include
        return result

    async def query(
        self, coll, query_embeddings, n_results, ids, where, where_document, include
    ) -> dict:
        include = _check_include(include)
        if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results < 1:
            raise InvalidArgumentError(
                f"Number of requested results {n_results}, cannot be negative, or zero."
            )
        if query_embeddings is None or len(query_embeddings) == 0:
            raise InvalidArgumentError("Expected query_embeddings to be a non-empty list")
        nq = len(query_embeddings)
        if coll.dimension is not None and query_embeddings.shape[1] != coll.dimension:
            raise InvalidArgumentError(
                f"Collection expecting embedding with dimension of {coll.dimension}, got {query_embeddings.shape[1]}"
            )
        groups: list[list[tuple]] = [[] for _ in range(nq)]
        cols = self._select_columns(include, prefix="r.")
        if coll.dimension is not None:
            cond, params = self._filters(ids, where, where_document)
            op, dist = _SPACE_SQL[coll.space]
            dist_sql = dist.format(q="q.v")
            plan = index_plan(coll.dimension, coll.space, self.info)
            order_sql = plan.order_by("q.v") if plan else f"embedding {op} q.v"
            filtered = bool(where) or bool(where_document) or ids is not None
            inner_cols = self._select_columns(include)
            query = sql.SQL(
                "SELECT q.ord, r.id, r.distance{extra} FROM unnest(%s::vector[]) WITH ORDINALITY AS q(v, ord) "
                "CROSS JOIN LATERAL (SELECT id, seq, {dist} AS distance{inner} FROM {t} WHERE {c} "
                "ORDER BY {order}, seq LIMIT %s) r ORDER BY q.ord, r.distance, r.seq"
            ).format(
                extra=sql.SQL("".join(", " + c for c in cols)),
                dist=sql.SQL(dist_sql),
                inner=sql.SQL("".join(", " + c for c in inner_cols)),
                t=coll.table,
                c=cond,
                order=sql.SQL(order_sql),
            )
            ef_search = int(coll.configuration.get("hnsw", {}).get("ef_search", 100))
            async with self._tx() as conn:
                # ef_search must cover n_results or HNSW returns fewer rows (max 1000).
                await conn.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(min(1000, max(ef_search, n_results))),),
                )
                if filtered:
                    if self.info.has_iterative_scan:
                        # pgvector 0.8: keep scanning the index until enough rows pass the filter.
                        await conn.execute(
                            "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
                        )
                    else:
                        # Older pgvector filters after the index scan and can return too few
                        # rows; fall back to exact search for filtered queries.
                        await conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
                rows = await (
                    await conn.execute(query, [list(query_embeddings), *params, n_results])
                ).fetchall()
            for row in rows:
                groups[row[0] - 1].append(row)
        result: dict[str, Any] = {"ids": [[r[1] for r in g] for g in groups]}
        result["distances"] = (
            [[float(r[2]) for r in g] for g in groups] if "distances" in include else None
        )
        pos = 3
        for item in ("embeddings", "documents", "metadatas", "uris"):
            if item in include:
                result[item] = [
                    [_as_array(r[pos]) if item == "embeddings" else r[pos] for r in g]
                    for g in groups
                ]
                pos += 1
            else:
                result[item] = None
        result["include"] = include
        return result
