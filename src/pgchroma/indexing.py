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
    """Build the HNSW index if the table has reached `threshold` rows. Returns True if it exists."""
    async with pool.connection() as conn:
        await conn.set_autocommit(True)
        try:
            row = await (
                await conn.execute(
                    "SELECT i.indisvalid FROM pg_index i "
                    "JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = %s AND c.relnamespace = current_schema()::regnamespace",
                    (index_name(table),),
                )
            ).fetchone()
            if row is not None and row[0]:
                return True
            n = (
                await (
                    await conn.execute(
                        sql.SQL("SELECT count(*) FROM (SELECT 1 FROM {} LIMIT %s) s").format(
                            sql.Identifier(table)
                        ),
                        (threshold,),
                    )
                ).fetchone()
            )[0]
            if n < threshold:
                return False
            locked = (
                await (
                    await conn.execute(
                        "SELECT pg_try_advisory_lock(hashtext(%s))", (f"pgchroma:index:{table}",)
                    )
                ).fetchone()
            )[0]
            if not locked:
                return False  # another process is building it
            try:
                if row is not None:  # an earlier build failed and left an invalid index
                    await conn.execute(
                        sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                            sql.Identifier(index_name(table))
                        )
                    )
                log.info("building HNSW index on %s (%s, %s)", table, plan.kind, plan.opclass)
                await conn.execute(
                    sql.SQL(
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS {} ON {} USING hnsw ({} {}) "
                        "WITH (m = {}, ef_construction = {})"
                    ).format(
                        sql.Identifier(index_name(table)),
                        sql.Identifier(table),
                        sql.SQL(plan.column()),
                        sql.SQL(plan.opclass),
                        sql.Literal(int(hnsw.get("max_neighbors", 16))),
                        sql.Literal(int(hnsw.get("ef_construction", 64))),
                    )
                )
                return True
            finally:
                await conn.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))", (f"pgchroma:index:{table}",)
                )
        finally:
            await conn.set_autocommit(False)
