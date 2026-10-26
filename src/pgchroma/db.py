"""Connection pool setup and catalog bootstrap."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources

import orjson
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import set_json_loads
from psycopg_pool import AsyncConnectionPool

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@dataclass(frozen=True)
class PgInfo:
    server_version: int
    pgvector_version: tuple[int, int, int]

    @property
    def has_iterative_scan(self) -> bool:
        return self.pgvector_version >= (0, 8, 0)

    @property
    def has_sparsevec(self) -> bool:
        return self.pgvector_version >= (0, 7, 0)

    @property
    def has_halfvec(self) -> bool:
        return self.pgvector_version >= (0, 7, 0)


def check_schema_name(schema: str) -> str:
    if not _SCHEMA_RE.match(schema):
        raise ValueError(f"invalid schema name {schema!r}: use letters, digits and underscores")
    return schema


def detect_transaction_pooler(database_url: str) -> bool:
    """True for URLs of transaction poolers, where session state does not survive a statement.

    Recognises Neon's pooled endpoints (`-pooler` hosts) and Supabase's transaction pooler
    (Supavisor on port 6543). Set PGCHROMA_POOLER=transaction for any other PgBouncer.
    """
    if os.environ.get("PGCHROMA_POOLER", "").lower() == "transaction":
        return True
    try:
        params = conninfo_to_dict(database_url)
    except psycopg.ProgrammingError:
        return False
    host, port = str(params.get("host") or ""), str(params.get("port") or "")
    return "-pooler." in host or port == "6543"


def _parse_version(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", text)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))  # type: ignore[return-value]


async def _prepare_database(database_url: str, schema: str) -> str:
    """Create the vector extension and the schema; return the extension's schema."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        row = await (
            await conn.execute(
                "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
                "WHERE e.extname = 'vector'"
            )
        ).fetchone()
        if row is None:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error as e:
                raise RuntimeError(
                    "pgvector is not installed in this database and could not be created "
                    f"({e}). Run `CREATE EXTENSION vector;` as a superuser, or enable the "
                    "'vector' extension in your provider's dashboard."
                ) from e
            row = await (
                await conn.execute(
                    "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
                    "ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
                )
            ).fetchone()
        await conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        return row[0]


async def open_pool(
    database_url: str,
    schema: str = "pgchroma",
    min_size: int = 1,
    max_size: int = 10,
) -> AsyncConnectionPool:
    """Open a pool whose connections use `schema` and know the pgvector types.

    The search_path is also set inside every transaction (see `use_schema`), because behind a
    transaction pooler the session-level setting below may belong to another server connection.
    """
    check_schema_name(schema)
    vector_schema = await _prepare_database(database_url, schema)
    search_path = ", ".join(
        '"' + s.replace('"', '""') + '"' for s in dict.fromkeys([schema, vector_schema, "public"])
    )

    async def configure(conn: psycopg.AsyncConnection) -> None:
        await conn.execute("SELECT set_config('search_path', %s, false)", (search_path,))
        await register_vector_async(conn)
        set_json_loads(orjson.loads, conn)
        await conn.commit()

    pool = AsyncConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"prepare_threshold": None},
        configure=configure,
        open=False,
    )
    await pool.open(wait=True)
    pool.pgchroma_schema = schema  # type: ignore[attr-defined]
    pool.pgchroma_search_path = search_path  # type: ignore[attr-defined]
    pool.pgchroma_pooled = detect_transaction_pooler(database_url)  # type: ignore[attr-defined]
    return pool


async def use_schema(conn: psycopg.AsyncConnection, pool: AsyncConnectionPool) -> None:
    """Set pgchroma's search_path for the current transaction only."""
    await conn.execute(
        "SELECT set_config('search_path', %s, true)",
        (pool.pgchroma_search_path,),  # type: ignore[attr-defined]
    )


async def bootstrap(pool: AsyncConnectionPool, schema: str) -> PgInfo:
    """Create catalog tables and default tenant/database. Safe to run concurrently."""
    ddl = resources.files("pgchroma").joinpath("catalog.sql").read_text()
    async with pool.connection() as conn:
        async with conn.transaction():
            await use_schema(conn, pool)
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"pgchroma:{schema}",)
            )
            await conn.execute(ddl)
        row = await (
            await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        info = PgInfo(conn.info.server_version, _parse_version(row[0]))
    pool.pgchroma_info = info  # type: ignore[attr-defined]
    return info
