"""HNSW index management.

Small collections are searched exactly (a sequential scan is fast and exact). Once a
collection reaches the threshold, an HNSW index is built with CREATE INDEX CONCURRENTLY in
the background, so writes keep flowing.

pgvector's `vector` HNSW index supports up to 2,000 dimensions. Larger embeddings (for example
OpenAI's 3,072-dimension models) are indexed as `halfvec` expressions on pgvector >= 0.7,
which supports up to 4,000 dimensions; otherwise those collections stay on exact search.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from .db import PgInfo

log = logging.getLogger("pgchroma")

_OPS = {"l2": "l2", "cosine": "cosine", "ip": "ip"}
_ORDER_OP = {"l2": "<->", "cosine": "<=>", "ip": "<#>"}


@dataclass(frozen=True)
class IndexPlan:
    kind: str  # "vector" or "halfvec"
    dimension: int
    space: str

    @property
    def opclass(self) -> str:
        return f"{self.kind}_{_OPS[self.space]}_ops"

    def column(self, col: str = "embedding") -> str:
        return col if self.kind == "vector" else f"({col}::halfvec({self.dimension}))"

    def order_by(self, query: str) -> str:
        """ORDER BY expression that matches the index expression, so the planner can use it."""
        q = query if self.kind == "vector" else f"{query}::halfvec({self.dimension})"
        col = "embedding" if self.kind == "vector" else f"embedding::halfvec({self.dimension})"
        return f"{col} {_ORDER_OP[self.space]} {q}"


def index_plan(dimension: int | None, space: str, info: PgInfo) -> IndexPlan | None:
    if dimension is None:
        return None
    if dimension <= 2000:
        return IndexPlan("vector", dimension, space)
    if dimension <= 4000 and info.has_halfvec:
        return IndexPlan("halfvec", dimension, space)
    return None


def index_name(table: str) -> str:
    return f"{table}_hnsw"


async def ensure_index(
    pool: AsyncConnectionPool,
    table: str,
    plan: IndexPlan,
    hnsw: dict,
    threshold: int,
) -> bool:
    """Build the HNSW index if the table has reached `threshold` rows. Returns True if it exists.

    CREATE INDEX CONCURRENTLY cannot run inside a transaction, so every statement here stands
    alone and names its schema: behind a transaction pooler each one may run on a different
    server connection. For the same reason, a lease row in `index_builds` (not a session
    advisory lock) keeps two processes from building the same index.
    """
    schema = pool.pgchroma_schema  # type: ignore[attr-defined]
    target = sql.Identifier(schema, table)
    builds = sql.Identifier(schema, "index_builds")
    regclass = target.as_string(None)
    owner = uuid.uuid4().hex
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        try:
            row = await (
                await conn.execute(
                    "SELECT i.indisvalid FROM pg_index i "
                    "JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = %s AND c.relnamespace = %s::regnamespace",
                    (index_name(table), sql.Identifier(schema).as_string(None)),
                )
            ).fetchone()
            if row is not None and row[0]:
                return True
            n = (
                await (
                    await conn.execute(
                        sql.SQL("SELECT count(*) FROM (SELECT 1 FROM {} LIMIT %s) s").format(
                            target
                        ),
                        (threshold,),
                    )
                ).fetchone()
            )[0]
            if n < threshold:
                return False
            # Take the lease unless another process holds a fresh one, or one whose build is
            # still running (pg_stat_progress_create_index shows it).
            claimed = await (
                await conn.execute(
                    sql.SQL(
                        "INSERT INTO {b} AS b (table_name, owner) VALUES (%s, %s) "
                        "ON CONFLICT (table_name) DO UPDATE SET owner = EXCLUDED.owner, "
                        "started_at = now() WHERE b.started_at < now() - interval '5 minutes' "
                        "AND NOT EXISTS (SELECT 1 FROM pg_stat_progress_create_index p "
                        "WHERE p.relid = %s::regclass) RETURNING owner"
                    ).format(b=builds),
                    (table, owner, regclass),
                )
            ).fetchone()
            if claimed is None:
                return False  # another process is building it
            try:
                if row is not None:  # an earlier build failed and left an invalid index
                    await conn.execute(
                        sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                            sql.Identifier(schema, index_name(table))
                        )
                    )
                log.info("building HNSW index on %s (%s, %s)", table, plan.kind, plan.opclass)
                if not pool.pgchroma_pooled:  # type: ignore[attr-defined]
                    # HNSW builds are much faster when the graph fits in maintenance_work_mem
                    # (Postgres defaults to 64MB). Session-level, so it needs no privileges;
                    # skipped behind a pooler, where it would stick to someone else's session.
                    await conn.execute(
                        "SELECT set_config('maintenance_work_mem', %s, false)",
                        (os.environ.get("PGCHROMA_MAINTENANCE_WORK_MEM", "1GB"),),
                    )
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON {} USING hnsw ({} {}) "
                        "WITH (m = {}, ef_construction = {})"
                    ).format(
                        sql.Identifier(index_name(table)),
                        target,
                        sql.SQL(plan.column()),
                        sql.SQL(plan.opclass),
                        sql.Literal(int(hnsw.get("max_neighbors", 16))),
                        sql.Literal(int(hnsw.get("ef_construction", 64))),
                    )
                )
                return True
            finally:
                await conn.execute(
                    sql.SQL("DELETE FROM {} WHERE table_name = %s AND owner = %s").format(builds),
                    (table, owner),
                )
        finally:
            await conn.set_autocommit(False)
