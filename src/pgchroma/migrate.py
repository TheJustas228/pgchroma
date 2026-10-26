"""Copy collections from a Chroma server into pgchroma's Postgres, resumably, and verify them."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import DuplicateError, NotFoundError
from .source import ChromaSource, SourceError
from .store import CollectionRef, Store

DEFAULT_TENANT = "default_tenant"

# Embedding functions that run on Chroma Cloud's servers and stop with it.
HOSTED_EMBEDDING_FUNCTIONS = (
    "chroma-cloud-qwen",
    "chroma-cloud-splade",
    "chroma_cloud_qwen",
    "chroma_cloud_splade",
)


class MigrationError(Exception):
    pass


@dataclass
class CollectionReport:
    name: str
    source_database: str
    target_database: str
    total: int = 0
    copied: int = 0
    resumed: bool = False
    indexed: bool = False
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class MigrationReport:
    collections: list[CollectionReport] = field(default_factory=list)


@dataclass
class CollectionCheck:
    name: str
    source_count: int
    target_count: int
    sampled: int = 0
    mismatched: list[str] = field(default_factory=list)
    query_overlap: float = 1.0


@dataclass
class VerifyReport:
    collections: list[CollectionCheck] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


Progress = Callable[[str, int, int], None]


def _source_key(source: ChromaSource, tenant: str, database: str, collection_id: str) -> str:
    return f"{source.url}|{tenant}|{database}|{collection_id}"


def _clean_metadata(m: Any) -> Any:
    # Chroma returns {} for records that only carry a uri; pgchroma reproduces that on read.
    return None if m == {} else m


LOCAL_QWEN = {
    "type": "known",
    "name": "sentence_transformer",
    "config": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "device": "cpu",
        "normalize_embeddings": True,
        "kwargs": {"model_kwargs": {"torch_dtype": "float32"}},
    },
}


LOCAL_SPLADE = {
    "type": "known",
    "name": "huggingface_sparse",
    "config": {
        "model_name": "prithivida/Splade_PP_en_v1",
        "device": "cpu",
        # Chroma Cloud encodes queries and documents the same way.
        "task": "document",
        "query_config": {"task": "document"},
        "kwargs": {},
    },
}
_LOCAL = {
    "chroma-cloud-qwen": LOCAL_QWEN,
    "chroma_cloud_qwen": LOCAL_QWEN,
    "chroma-cloud-splade": LOCAL_SPLADE,
    "chroma_cloud_splade": LOCAL_SPLADE,
}


def _embedding_functions(node: Any) -> list[dict]:
    """Every embedding function config in a collection configuration or schema."""
    if isinstance(node, list):
        return [ef for item in node for ef in _embedding_functions(item)]
    if not isinstance(node, dict):
        return []
    found = []
    for key, value in node.items():
        if key == "embedding_function" and isinstance(value, dict):
            found.append(value)
        else:
            found.extend(_embedding_functions(value))
    return found


def _replace_hosted(node: Any) -> Any:
    if isinstance(node, list):
        return [_replace_hosted(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "embedding_function" and isinstance(value, dict) and value.get("name") in _LOCAL:
            out[key] = copy.deepcopy(_LOCAL[value["name"]])
        else:
            out[key] = _replace_hosted(value)
    return out


def localize_embedding_functions(
    configuration: dict | None, schema: dict | None
) -> tuple[dict | None, dict | None, list[str]]:
    """Swap Chroma Cloud's hosted embedding functions for the same open models run locally.

    `chroma-cloud-qwen` is Qwen/Qwen3-Embedding-0.6B and `chroma-cloud-splade` is
    prithivida/Splade_PP_en_v1 (see chromadb's ChromaCloud*EmbeddingFunction). Stored embeddings
    are untouched; this only changes what the client uses for new documents and query text.
    Returns the new configuration and schema, with a note for every function replaced.
    """
    hosted = {
        ef["name"].replace("_", "-"): ef
        for ef in _embedding_functions(configuration) + _embedding_functions(schema)
        if ef.get("name") in _LOCAL
    }
    notes = []
    if qwen := hosted.get("chroma-cloud-qwen"):
        note = (
            "embedding function chroma-cloud-qwen replaced with the same open model run locally "
            "(Qwen/Qwen3-Embedding-0.6B via sentence-transformers; `pip install sentence-transformers`)"
        )
        if task := (qwen.get("config") or {}).get("task"):
            note += (
                f". Its queries used the '{task}' instruction: use the embedding function in "
                "examples/local_qwen_embeddings.py to reproduce query embeddings exactly"
            )
        notes.append(note)
    if "chroma-cloud-splade" in hosted:
        notes.append(
            "sparse embedding function chroma-cloud-splade replaced with the same open model run "
            "locally (prithivida/Splade_PP_en_v1 via sentence-transformers; "
            "`pip install sentence-transformers`)"
        )
    if not notes:
        return configuration, schema, []
    return _replace_hosted(configuration), _replace_hosted(schema), notes


def hosted_embedding_functions(collection: dict) -> list[str]:
    efs = _embedding_functions(collection.get("configuration_json")) + _embedding_functions(
        collection.get("schema")
    )
    return sorted({ef["name"] for ef in efs if ef.get("name") in HOSTED_EMBEDDING_FUNCTIONS})


async def _checkpoint(
    store: Store, key: str, collection_id, *, copied: int | None = None, done: bool = False
):
    async with store._tx() as conn:
        if copied is None:
            row = await (
                await conn.execute(
                    "SELECT copied, done FROM migrations WHERE source = %s AND collection_id = %s",
                    (key, collection_id),
                )
            ).fetchone()
            return row
        await conn.execute(
            "INSERT INTO migrations (source, collection_id, copied, done) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (source, collection_id) DO UPDATE SET copied = EXCLUDED.copied, "
            "done = EXCLUDED.done, updated_at = now()",
            (key, collection_id, copied, done),
        )
        return None


async def _ensure_target(store: Store, tenant: str, database: str) -> None:
    with contextlib.suppress(DuplicateError):
        await store.create_tenant(tenant)
    with contextlib.suppress(DuplicateError):
        await store.create_database(tenant, database)


async def _target_collection(
    store: Store,
    source: ChromaSource,
    key: str,
    coll: dict,
    to_tenant: str,
    to_db: str,
    configuration: dict | None = None,
    schema: dict | None = None,
) -> tuple[CollectionRef, int, bool]:
    """Return (target collection, records already copied, resumed?)."""
    try:
        existing = await store.get_collection(to_tenant, to_db, coll["name"])
    except NotFoundError:
        existing = None
    if existing is not None:
        ref = await store.collection(to_tenant, to_db, existing["id"])
        row = await _checkpoint(store, key, ref.id)
        if row is None:
            raise MigrationError(
                f"Collection [{coll['name']}] already exists in database [{to_db}] and was not "
                "created by this migration. Delete it, or migrate into another database with --to-database."
            )
        return ref, int(row[0]), True
    created = await store.create_collection(
        to_tenant,
        to_db,
        coll["name"],
        coll.get("metadata"),
        coll.get("configuration_json") if configuration is None else configuration,
        coll.get("schema") if schema is None else schema,
        False,
    )
    ref = await store.collection(to_tenant, to_db, created["id"])
    await _checkpoint(store, key, ref.id, copied=0)
    return ref, 0, False


async def _copy_collection(
    source: ChromaSource,
    store: Store,
    tenant: str,
    database: str,
    coll: dict,
    to_tenant: str,
    to_db: str,
    batch_size: int,
    progress: Progress | None,
    build_indexes: bool = True,
    local_embedding_functions: bool = False,
) -> CollectionReport:
    started = time.monotonic()
    report = CollectionReport(coll["name"], database, to_db)
    configuration, schema = coll.get("configuration_json"), coll.get("schema")
    notes = []
    if local_embedding_functions:
        configuration, schema, notes = localize_embedding_functions(configuration, schema)
        report.warnings.extend(notes)
    if not notes:
        for ef in hosted_embedding_functions(coll):
            report.warnings.append(
                f"uses the Chroma-hosted embedding function '{ef}', which stops working with "
                "Chroma Cloud; rerun with --local-embedding-functions or see the README"
            )
    key = _source_key(source, tenant, database, coll["id"])
    ref, offset, report.resumed = await _target_collection(
        store, source, key, coll, to_tenant, to_db, configuration, schema
    )
    report.total = await source.count(database, coll["id"])
    size = batch_size
    if progress:
        progress(coll["name"], offset, report.total)
    while True:
        try:
            page = await source.get_page(database, coll["id"], offset, size)
        except SourceError as e:
            if e.is_size_limit and size > 1:
                size = max(1, size // 2)
                continue
            raise
        ids = page["ids"]
        if not ids:
            break
        await store.add(
            ref,
            ids,
            np.asarray(page["embeddings"], dtype=np.float32),
            [_clean_metadata(m) for m in page.get("metadatas") or [None] * len(ids)],
            page.get("documents"),
            page.get("uris"),
        )
        offset += len(ids)
        await _checkpoint(store, key, ref.id, copied=offset)
        if progress:
            progress(coll["name"], offset, report.total)
        if len(ids) < size:
            break
    await _checkpoint(store, key, ref.id, copied=offset, done=True)
    # Copy first, index once: building HNSW after the bulk load is far faster than
    # maintaining it during inserts. Small collections stay on exact search.
    if build_indexes:
        report.indexed = await store.build_index(ref, force=False)
    report.copied = offset
    report.seconds = time.monotonic() - started
    return report


def _plan(source_dbs: list[str], to_database: str | None) -> dict[str, str]:
    if to_database is None:
        return {db: db for db in source_dbs}
    return {db: to_database for db in source_dbs}


async def _jobs(source: ChromaSource, databases, collections) -> list[tuple[str, dict]]:
    dbs = databases or await source.databases()
    jobs = []
    for db in dbs:
        for coll in await source.collections(db):
            if collections is None or coll["name"] in collections:
                jobs.append((db, coll))
    if collections is not None:
        missing = set(collections) - {c["name"] for _, c in jobs}
        if missing:
            raise MigrationError(
                f"Collections not found in the source: {', '.join(sorted(missing))}"
            )
    return jobs


async def migrate(
    source: ChromaSource,
    store: Store,
    *,
    databases: list[str] | None = None,
    collections: list[str] | None = None,
    to_tenant: str = DEFAULT_TENANT,
    to_database: str | None = None,
    batch_size: int = 300,
    concurrency: int = 4,
    progress: Progress | None = None,
    build_indexes: bool = True,
    local_embedding_functions: bool = False,
) -> MigrationReport:
    tenant = await source.resolve_tenant()
    jobs = await _jobs(source, databases, collections)
    mapping = _plan(sorted({db for db, _ in jobs}), to_database)
    seen: dict[tuple[str, str], str] = {}
    for db, coll in jobs:
        target = (mapping[db], coll["name"])
        if target in seen and seen[target] != db:
            raise MigrationError(
                f"Collection [{coll['name']}] exists in source databases [{seen[target]}] and [{db}]; "
                f"they cannot both be migrated into [{mapping[db]}]."
            )
        seen[target] = db
    for target_db in set(mapping.values()):
        await _ensure_target(store, to_tenant, target_db)

    sem = asyncio.Semaphore(max(1, concurrency))

    async def run(db: str, coll: dict) -> CollectionReport:
        async with sem:
            return await _copy_collection(
                source,
                store,
                tenant,
                db,
                coll,
                to_tenant,
                mapping[db],
                batch_size,
                progress,
                build_indexes,
                local_embedding_functions,
            )

    tasks = [asyncio.ensure_future(run(db, coll)) for db, coll in jobs]
    try:
        # Stop everything on the first failure so a retry never races leftover work.
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            if t.exception() is not None:
                raise t.exception()
        return MigrationReport([t.result() for t in tasks])
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _same_record(src: dict, dst: dict, i: int, j: int) -> list[str]:
    problems = []
    if src["documents"][i] != dst["documents"][j]:
        problems.append("document")
    if src["uris"][i] != dst["uris"][j]:
        problems.append("uri")
    if (src["metadatas"][i] or None) != (dst["metadatas"][j] or None):
        problems.append("metadata")
    a = np.asarray(src["embeddings"][i], dtype=np.float32)
    b = np.asarray(dst["embeddings"][j], dtype=np.float32)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-7):
        problems.append("embedding")
    return problems


def _distance(space: str, q: np.ndarray, e: np.ndarray) -> float:
    q, e = q.astype(np.float64), e.astype(np.float64)
    if space == "cosine":
        return 1.0 - float(q @ e) / max(float(np.linalg.norm(q) * np.linalg.norm(e)), 1e-30)
    if space == "ip":
        return 1.0 - float(q @ e)
    return float(((q - e) ** 2).sum())


async def _query_overlap(store: Store, ref, queries: np.ndarray, theirs: list, ours: dict) -> float:
    """Share of the source's results that are exact nearest neighbours.

    A result counts when pgchroma's exact search returned it, or when it is no farther from the
    query than pgchroma's k-th neighbour: duplicates and other ties may come back in any order.
    """
    others = sorted({i for t, o in zip(theirs, ours["ids"], strict=True) for i in set(t) - set(o)})
    found = {}
    if others:
        got = await store.get(ref, others, None, None, None, None, ["embeddings"])
        found = dict(zip(got["ids"], got["embeddings"], strict=True))
    overlaps = []
    for qv, t, o, dists in zip(queries, theirs, ours["ids"], ours["distances"], strict=True):
        kth = max(dists) if dists else 0.0
        tol = 1e-4 * max(1.0, abs(kth)) + 1e-6
        mine = set(o)
        hits = sum(
            1
            for i in t
            if i in mine or (i in found and _distance(ref.space, qv, found[i]) <= kth + tol)
        )
        overlaps.append(hits / max(1, len(t)))
    return float(np.mean(overlaps))


async def verify(
    source: ChromaSource,
    store: Store,
    *,
    databases: list[str] | None = None,
    collections: list[str] | None = None,
    to_tenant: str = DEFAULT_TENANT,
    to_database: str | None = None,
    sample: int = 50,
    k: int = 10,
    seed: int = 0,
) -> VerifyReport:
    await source.resolve_tenant()
    jobs = await _jobs(source, databases, collections)
    mapping = _plan(sorted({db for db, _ in jobs}), to_database)
    rng = random.Random(seed)
    report = VerifyReport()
    for db, coll in jobs:
        name = coll["name"]
        try:
            target = await store.get_collection(to_tenant, mapping[db], name)
        except NotFoundError:
            report.problems.append(f"{name}: missing in the target")
            continue
        ref = await store.collection(to_tenant, mapping[db], target["id"])
        check = CollectionCheck(name, await source.count(db, coll["id"]), await store.count(ref))
        report.collections.append(check)
        if check.source_count != check.target_count:
            report.problems.append(
                f"{name}: {check.source_count} records in the source, {check.target_count} in the target"
            )
        if check.source_count == 0:
            continue
        # Sample a few random pages from the source and compare record by record.
        chunk = max(1, min(sample, check.source_count) // 4)
        src_pages = []
        for _ in range(4):
            offset = rng.randrange(0, max(1, check.source_count - chunk + 1))
            src_pages.append(await source.get_page(db, coll["id"], offset, chunk))
        sampled_embeddings = []
        for page in src_pages:
            if not page["ids"]:
                continue
            mine = await store.get(
                ref,
                page["ids"],
                None,
                None,
                None,
                None,
                ["embeddings", "documents", "metadatas", "uris"],
            )
            pos = {rid: j for j, rid in enumerate(mine["ids"])}
            for i, rid in enumerate(page["ids"]):
                check.sampled += 1
                if rid not in pos:
                    check.mismatched.append(f"{rid}: missing")
                    continue
                diff = _same_record(page, mine, i, pos[rid])
                if diff:
                    check.mismatched.append(f"{rid}: {', '.join(diff)} differ")
            sampled_embeddings.extend(page["embeddings"][:3])
        if check.mismatched:
            report.problems.append(
                f"{name}: {len(check.mismatched)} sampled records differ, e.g. {check.mismatched[0]}"
            )
        # Query parity: the same queries should find (nearly) the same neighbours.
        queries = sampled_embeddings[:10]
        if queries:
            n = min(k, check.source_count)
            theirs = await source.query(db, coll["id"], queries, n)
            # Compare with pgchroma's exact search: the source may use an approximate index, and
            # comparing two approximate answers would understate agreement.
            q = np.asarray(queries, dtype=np.float32)
            ours = await store.query(ref, q, n, None, None, None, ["distances"], exact=True)
            check.query_overlap = await _query_overlap(store, ref, q, theirs["ids"], ours)
            if check.query_overlap < 0.5:
                report.problems.append(
                    f"{name}: only {check.query_overlap:.0%} of the source's top-{n} results are the "
                    "exact nearest neighbours here; check the collection's distance space"
                )
            elif check.query_overlap < 0.9:
                report.warnings.append(
                    f"{name}: the source's top-{n} results agree {check.query_overlap:.0%} with exact "
                    "search; this is normal for an approximate (HNSW/SPANN) source index"
                )
    return report
