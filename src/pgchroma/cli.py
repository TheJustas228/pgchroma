"""Command line: `pgchroma serve | migrate | verify | doctor`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__

console = Console(highlight=False, soft_wrap=True)

ENV_HELP = (
    "Environment variables: DATABASE_URL, PGCHROMA_SCHEMA, PGCHROMA_AUTH_TOKENS (comma separated), "
    "CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE, HOST, PORT."
)


def _env_list(name: str) -> list[str]:
    return [t.strip() for t in os.environ.get(name, "").split(",") if t.strip()]


def _require_database_url(args) -> str:
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        console.print(
            "[red]error:[/] pass --database-url or set DATABASE_URL (a Postgres connection string)"
        )
        raise SystemExit(2)
    return url


def _add_db_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--database-url", help="Postgres connection string (default: $DATABASE_URL)")
    p.add_argument(
        "--schema",
        default=os.environ.get("PGCHROMA_SCHEMA", "pgchroma"),
        help="Postgres schema that holds pgchroma's tables (default: pgchroma)",
    )


def _add_source_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("source (Chroma Cloud or any Chroma >= 1.0 server)")
    src.add_argument(
        "--from", dest="source_url", help="Chroma server URL, e.g. http://localhost:8000"
    )
    src.add_argument(
        "--from-cloud", action="store_true", help="read from Chroma Cloud (api.trychroma.com)"
    )
    src.add_argument(
        "--api-key",
        default=os.environ.get("CHROMA_API_KEY"),
        help="Chroma API key (default: $CHROMA_API_KEY)",
    )
    src.add_argument(
        "--tenant",
        default=os.environ.get("CHROMA_TENANT"),
        help="source tenant (default: from the API key)",
    )
    src.add_argument(
        "--database",
        action="append",
        dest="databases",
        help="source database (repeatable; default: all)",
    )
    src.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="collection name (repeatable; default: all)",
    )
    dst = p.add_argument_group("target")
    dst.add_argument(
        "--to-tenant", default="default_tenant", help="target tenant (default: default_tenant)"
    )
    dst.add_argument(
        "--to-database", help="target database (default: same name as the source database)"
    )


def _source(args):
    from .source import CHROMA_CLOUD_URL, ChromaSource

    if not args.source_url and not args.from_cloud:
        console.print("[red]error:[/] pass --from <url> or --from-cloud")
        raise SystemExit(2)
    url = CHROMA_CLOUD_URL if args.from_cloud else args.source_url
    if args.from_cloud and not args.api_key:
        console.print("[red]error:[/] Chroma Cloud needs --api-key (or CHROMA_API_KEY)")
        raise SystemExit(2)
    single_db = args.databases[0] if args.databases and len(args.databases) == 1 else None
    return ChromaSource(url, api_key=args.api_key, tenant=args.tenant, database=single_db)


async def _open_store(database_url: str, schema: str):
    from .db import bootstrap, open_pool
    from .store import Store

    pool = await open_pool(database_url, schema, min_size=1, max_size=8)
    info = await bootstrap(pool, schema)
    return Store(pool, info)


# ----------------------------------------------------------------- serve


def cmd_serve(args) -> int:
    import uvicorn

    os.environ["DATABASE_URL"] = _require_database_url(args)
    os.environ["PGCHROMA_SCHEMA"] = args.schema
    if args.auth_token:
        os.environ["PGCHROMA_AUTH_TOKENS"] = ",".join(args.auth_token)
    if args.allow_reset:
        os.environ["PGCHROMA_ALLOW_RESET"] = "1"
    os.environ["PGCHROMA_INDEX_THRESHOLD"] = str(args.index_threshold)
    os.environ["PGCHROMA_MAX_BATCH_SIZE"] = str(args.max_batch_size)
    os.environ["PGCHROMA_POOL_SIZE"] = str(args.pool_size)
    console.print(
        f"[bold]pgchroma {__version__}[/] serving the Chroma API on http://{args.host}:{args.port} "
        f"(schema [cyan]{args.schema}[/], auth {'on' if args.auth_token or _env_list('PGCHROMA_AUTH_TOKENS') else 'off'})"
    )
    uvicorn.run(
        "pgchroma.server:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        proxy_headers=True,
    )
    return 0


# --------------------------------------------------------------- migrate


def _print_verify(report) -> bool:
    table = Table(title="Verification", title_justify="left")
    for col in ("collection", "source", "target", "sampled", "differ", "top-k vs exact"):
        table.add_column(col, justify="right" if col != "collection" else "left")
    for c in report.collections:
        table.add_row(
            c.name,
            str(c.source_count),
            str(c.target_count),
            str(c.sampled),
            str(len(c.mismatched)),
            f"{c.query_overlap:.0%}",
        )
    console.print(table)
    for w in report.warnings:
        console.print(f"[yellow]![/] {w}")
    if report.ok:
        console.print("[green]✔ Verified:[/] counts match and every sampled record is identical.")
        return True
    for p in report.problems:
        console.print(f"[red]✘[/] {p}")
    return False


async def _migrate(args) -> int:
    from .migrate import MigrationError, migrate, verify
    from .source import SourceError

    store = await _open_store(_require_database_url(args), args.schema)
    try:
        async with _source(args) as src:
            tenant = await src.resolve_tenant()
            console.print(f"Source: [cyan]{src.url}[/] tenant [cyan]{tenant}[/]")
            started = time.monotonic()
            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as bar:
                tasks: dict[str, int] = {}

                def progress(name: str, done: int, total: int) -> None:
                    if name not in tasks:
                        tasks[name] = bar.add_task(name, total=total or None)
                    bar.update(tasks[name], completed=done, total=total or None)

                report = await migrate(
                    src,
                    store,
                    databases=args.databases,
                    collections=args.collections,
                    to_tenant=args.to_tenant,
                    to_database=args.to_database,
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    progress=progress,
                    local_embedding_functions=args.local_embedding_functions,
                )
            table = Table(title="Migrated", title_justify="left")
            for col in (
                "collection",
                "from database",
                "to database",
                "records",
                "seconds",
                "index",
                "resumed",
            ):
                table.add_column(
                    col,
                    justify="left"
                    if col in ("collection", "from database", "to database")
                    else "right",
                )
            for c in report.collections:
                table.add_row(
                    c.name,
                    c.source_database,
                    c.target_database,
                    f"{c.copied:,}",
                    f"{c.seconds:.1f}",
                    "hnsw" if c.indexed else "exact",
                    "yes" if c.resumed else "",
                )
            console.print(table)
            total = sum(c.copied for c in report.collections)
            console.print(f"Copied {total:,} records in {time.monotonic() - started:.1f}s.")
            for c in report.collections:
                for w in c.warnings:
                    console.print(f"[yellow]![/] {c.name}: {w}")
            ok = True
            if not args.no_verify:
                check = await verify(
                    src,
                    store,
                    databases=args.databases,
                    collections=args.collections,
                    to_tenant=args.to_tenant,
                    to_database=args.to_database,
                    sample=args.sample,
                )
                ok = _print_verify(check)
            dbs = sorted({c.target_database for c in report.collections})
            hint_db = dbs[0] if dbs else "default_database"
            console.print(
                "\nNext: start the server ([bold]pgchroma serve[/]) and point your client at it:\n"
                f'  chromadb.HttpClient(host="localhost", port=8000, tenant="{args.to_tenant}", database="{hint_db}")'
            )
            return 0 if ok else 1
    except (MigrationError, SourceError) as e:
        console.print(f"[red]error:[/] {e}")
        return 1
    finally:
        await store.wait_for_indexing()
        await store.pool.close()


def cmd_migrate(args) -> int:
    return asyncio.run(_migrate(args))


async def _verify(args) -> int:
    from .migrate import MigrationError, verify
    from .source import SourceError

    store = await _open_store(_require_database_url(args), args.schema)
    try:
        async with _source(args) as src:
            report = await verify(
                src,
                store,
                databases=args.databases,
                collections=args.collections,
                to_tenant=args.to_tenant,
                to_database=args.to_database,
                sample=args.sample,
            )
            return 0 if _print_verify(report) else 1
    except (MigrationError, SourceError) as e:
        console.print(f"[red]error:[/] {e}")
        return 1
    finally:
        await store.pool.close()


def cmd_verify(args) -> int:
    return asyncio.run(_verify(args))


# ---------------------------------------------------------------- doctor


async def _doctor(args) -> int:
    import psycopg

    from .indexing import index_name, index_plan

    url = _require_database_url(args)
    table = Table(show_header=False)
    table.add_column("check")
    table.add_column("result")
    ok = True
    try:
        store = await _open_store(url, args.schema)
    except (psycopg.Error, RuntimeError) as e:
        console.print(f"[red]✘ cannot use the database:[/] {e}")
        return 1
    try:
        info = store.info
        major = info.server_version // 10000
        table.add_row(
            "PostgreSQL",
            f"{major}.{info.server_version % 10000}"
            + ("" if major >= 14 else "  [red](14+ required)[/]"),
        )
        pv = ".".join(map(str, info.pgvector_version))
        note = (
            ""
            if info.has_iterative_scan
            else "  [yellow](0.8+ recommended: faster filtered queries; 0.7+ indexes >2000-dim vectors)[/]"
        )
        table.add_row("pgvector", pv + note)
        table.add_row("schema", args.schema)
        async with store.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT c.id, c.name, d.tenant, d.name, c.dimension, c.space FROM collections c "
                    "JOIN databases d ON d.id = c.database_id ORDER BY d.tenant, d.name, c.name"
                )
            ).fetchall()
            details = []
            for cid, name, tenant, db, dim, space in rows:
                t = f"c_{cid.hex}"
                n = (await (await conn.execute(f'SELECT count(*) FROM "{t}"')).fetchone())[0]
                has_index = (
                    await (
                        await conn.execute(
                            "SELECT 1 FROM pg_class WHERE relname = %s AND relnamespace = current_schema()::regnamespace",
                            (index_name(t),),
                        )
                    ).fetchone()
                ) is not None
                plan = index_plan(dim, space, info)
                if has_index:
                    idx = "hnsw"
                elif plan is None and dim:
                    idx = "exact only"
                elif n >= store.index_threshold:
                    idx = "[yellow]pending (built on next write)[/]"
                else:
                    idx = "exact (small)"
                details.append((f"{tenant}/{db}/{name}", n, dim, space, idx))
        table.add_row("collections", str(len(rows)))
        console.print(table)
        if details:
            t2 = Table("collection", "records", "dim", "space", "index")
            for row in details:
                t2.add_row(row[0], f"{row[1]:,}", str(row[2] or "-"), row[3], row[4])
            console.print(t2)
    finally:
        await store.pool.close()
    console.print("[green]✔ ready[/]" if ok else "[red]✘ problems found[/]")
    return 0 if ok else 1


def cmd_doctor(args) -> int:
    return asyncio.run(_doctor(args))


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgchroma",
        description="A Chroma-compatible server that stores everything in Postgres + pgvector.",
        epilog=ENV_HELP,
    )
    parser.add_argument("--version", action="version", version=f"pgchroma {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the Chroma-compatible HTTP server", epilog=ENV_HELP)
    _add_db_args(s)
    s.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--workers", type=int, default=1, help="uvicorn worker processes")
    s.add_argument(
        "--auth-token",
        action="append",
        help="require this token (repeatable); clients send it as x-chroma-token",
    )
    s.add_argument(
        "--allow-reset", action="store_true", help="allow POST /api/v2/reset (deletes everything)"
    )
    s.add_argument(
        "--index-threshold",
        type=int,
        default=int(os.environ.get("PGCHROMA_INDEX_THRESHOLD", "10000")),
        help="build an HNSW index once a collection has this many records",
    )
    s.add_argument(
        "--max-batch-size", type=int, default=int(os.environ.get("PGCHROMA_MAX_BATCH_SIZE", "5461"))
    )
    s.add_argument(
        "--pool-size",
        type=int,
        default=int(os.environ.get("PGCHROMA_POOL_SIZE", "10")),
        help="max Postgres connections per worker",
    )
    s.add_argument("--log-level", default="info")
    s.set_defaults(fn=cmd_serve)

    m = sub.add_parser(
        "migrate",
        help="copy collections from Chroma Cloud or a Chroma server, then verify",
        epilog=ENV_HELP,
    )
    _add_source_args(m)
    _add_db_args(m)
    m.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="records per request (halved automatically on quota errors)",
    )
    m.add_argument("--concurrency", type=int, default=4, help="collections copied in parallel")
    m.add_argument(
        "--sample", type=int, default=50, help="records per collection compared during verification"
    )
    m.add_argument("--no-verify", action="store_true", help="skip verification")
    m.add_argument(
        "--local-embedding-functions",
        action="store_true",
        help="replace Chroma Cloud's hosted chroma-cloud-qwen with the same open model run locally",
    )
    m.set_defaults(fn=cmd_migrate)

    v = sub.add_parser("verify", help="compare pgchroma with the source again", epilog=ENV_HELP)
    _add_source_args(v)
    _add_db_args(v)
    v.add_argument("--sample", type=int, default=50)
    v.set_defaults(fn=cmd_verify)

    d = sub.add_parser(
        "doctor", help="check Postgres, pgvector and collection indexes", epilog=ENV_HELP
    )
    _add_db_args(d)
    d.set_defaults(fn=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
