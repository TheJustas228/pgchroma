"""Save collections from a Chroma server to JSON Lines files, and read such an export back.

`export` writes one file per collection: a header line with the collection as Chroma lists it,
then one line per record. It only reads from the source, so it works during Chroma Cloud's
read-only month, and it resumes where it stopped. `DirectorySource` serves an export through the
same read calls as `ChromaSource`, so `migrate` and `verify` load it into Postgres unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import IO, Any
from urllib.parse import quote

import numpy as np

from .config import normalize_configuration
from .migrate import MigrationError, Progress, _jobs, hosted_embedding_functions, run_all
from .source import ChromaSource, SourceError

MANIFEST = "manifest.json"
FORMAT = "pgchroma-export/1"
FIELDS = (("embedding", "embeddings"), ("document", "documents"), ("metadata", "metadatas"))


@dataclass
class CollectionExport:
    name: str
    database: str
    path: Path
    records: int = 0
    source_count: int = 0
    resumed: bool = False
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExportReport:
    collections: list[CollectionExport] = field(default_factory=list)


def _load_manifest(out: Path) -> dict | None:
    try:
        return json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _save_manifest(out: Path, manifest: dict) -> None:
    tmp = out / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out / MANIFEST)


def _resume_point(path: Path, collection_id: str) -> int | None:
    """Records already in `path`, after dropping a half-written last line; None to start over."""
    if not path.exists():
        return None
    with path.open("rb+") as f:
        first = f.readline()
        try:
            if not first.endswith(b"\n") or json.loads(first)["collection"]["id"] != collection_id:
                return None
        except (ValueError, KeyError, TypeError):
            return None
        lines, pos = 0, f.tell()
        end = pos
        while chunk := f.read(1 << 20):
            pos += len(chunk)
            if (last := chunk.rfind(b"\n")) >= 0:
                lines += chunk.count(b"\n")
                end = pos - len(chunk) + last + 1
        f.truncate(end)
    return lines


def _record_lines(page: dict) -> str:
    ids = page["ids"]
    columns = {
        key: page.get(plural) or [None] * len(ids) for key, plural in (*FIELDS, ("uri", "uris"))
    }
    return "".join(
        json.dumps({"id": rid, **{key: col[i] for key, col in columns.items()}}, ensure_ascii=False)
        + "\n"
        for i, rid in enumerate(ids)
    )


async def _export_collection(
    source: ChromaSource,
    out: Path,
    manifest: dict,
    rel: str,
    database: str,
    coll: dict,
    batch_size: int,
    progress: Progress | None,
) -> CollectionExport:
    started = time.monotonic()
    entry = manifest["collections"][rel]
    path = out / rel
    report = CollectionExport(coll["name"], database, path)
    report.source_count = await source.count(database, coll["id"])
    for ef in hosted_embedding_functions(coll):
        report.warnings.append(
            f"uses the Chroma-hosted embedding function '{ef}'; the stored embeddings are saved, "
            "and `pgchroma migrate --from <export> --local-embedding-functions` sets up the same "
            "open model for new ones"
        )
    if entry["done"] and path.exists():
        report.records, report.resumed = entry["records"], True
        if progress:
            progress(coll["name"], report.records, report.records)
        return report

    path.parent.mkdir(parents=True, exist_ok=True)
    offset = _resume_point(path, coll["id"])
    report.resumed = offset is not None
    if offset is None:
        line = json.dumps({"collection": coll}, ensure_ascii=False) + "\n"
        path.write_text(line, encoding="utf-8", newline="\n")
        offset = 0
    size = batch_size
    if progress:
        progress(coll["name"], offset, report.source_count)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        while True:
            try:
                page = await source.get_page(database, coll["id"], offset, size)
            except SourceError as e:
                if e.is_size_limit and size > 1:
                    size = max(1, size // 2)
                    continue
                raise
            if not page["ids"]:
                break
            f.write(_record_lines(page))
            f.flush()
            offset += len(page["ids"])
            if progress:
                progress(coll["name"], offset, report.source_count)
            if len(page["ids"]) < size:
                break
    entry["records"], entry["done"] = offset, True
    _save_manifest(out, manifest)
    if offset != report.source_count:
        report.warnings.append(
            f"the source reported {report.source_count} records and {offset} were saved; "
            "it may have changed during the export"
        )
    report.records = offset
    report.seconds = time.monotonic() - started
    return report


async def export(
    source: ChromaSource,
    out: str | Path,
    *,
    databases: list[str] | None = None,
    collections: list[str] | None = None,
    batch_size: int = 300,
    concurrency: int = 4,
    progress: Progress | None = None,
) -> ExportReport:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tenant = await source.resolve_tenant()
    manifest = _load_manifest(out) or {
        "format": FORMAT,
        "source": source.url,
        "tenant": tenant,
        "collections": {},
    }
    if (manifest.get("source"), manifest.get("tenant")) != (source.url, tenant):
        raise MigrationError(
            f"{out} already holds an export of {manifest.get('source')} (tenant "
            f"{manifest.get('tenant')}); choose another --out directory"
        )
    jobs = []
    for db, coll in await _jobs(source, databases, collections):
        rel = f"{quote(db, safe='')}/{quote(coll['name'], safe='')}.jsonl"
        entry = manifest["collections"].get(rel)
        if entry is None or entry["id"] != coll["id"]:
            manifest["collections"][rel] = {
                "database": db,
                "name": coll["name"],
                "id": coll["id"],
                "records": 0,
                "done": False,
            }
        jobs.append((rel, db, coll))
    # Register every collection before copying any, so a reader can tell the export is
    # unfinished even if it stopped before reaching some of them.
    _save_manifest(out, manifest)

    sem = asyncio.Semaphore(max(1, concurrency))

    async def run(rel: str, db: str, coll: dict) -> CollectionExport:
        async with sem:
            return await _export_collection(
                source, out, manifest, rel, db, coll, batch_size, progress
            )

    return ExportReport(await run_all([run(*job) for job in jobs]))


def _distances(space: str, queries: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    q, e = queries.astype(np.float64), embeddings.astype(np.float64)
    dots = q @ e.T
    if space == "cosine":
        norms = np.outer(np.linalg.norm(q, axis=1), np.linalg.norm(e, axis=1))
        return 1.0 - dots / np.maximum(norms, 1e-30)
    if space == "ip":
        return 1.0 - dots
    return (q * q).sum(1)[:, None] + (e * e).sum(1)[None, :] - 2.0 * dots


class DirectorySource:
    """Reads an export written by `export` with the read calls `migrate` and `verify` make."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.url = self.path.as_uri()
        self.tenant: str | None = None
        self._entries: dict[tuple[str, str], dict] | None = None
        self._headers: dict[str, dict] = {}
        self._readers: dict[str, tuple[IO[str], int]] = {}

    async def __aenter__(self) -> DirectorySource:
        return self

    async def __aexit__(self, *exc) -> None:
        for f, _ in self._readers.values():
            f.close()
        self._readers.clear()

    def _manifest(self) -> dict[tuple[str, str], dict]:
        if self._entries is None:
            manifest = _load_manifest(self.path)
            if manifest is None or manifest.get("format") != FORMAT:
                raise SourceError(404, f"{self.path} is not a pgchroma export (no {MANIFEST})")
            unfinished = sorted(
                e["name"] for e in manifest["collections"].values() if not e["done"]
            )
            if unfinished:
                raise SourceError(
                    409,
                    f"the export in {self.path} is incomplete ({', '.join(unfinished)} unfinished); "
                    "run the same pgchroma export command again to finish it",
                )
            self.tenant = manifest["tenant"]
            self._entries = {
                (e["database"], e["id"]): {**e, "file": rel}
                for rel, e in manifest["collections"].items()
            }
        return self._entries

    def _entry(self, database: str, collection_id: str) -> dict:
        try:
            return self._manifest()[(database, collection_id)]
        except KeyError:
            raise SourceError(404, f"collection {collection_id} is not in the export") from None

    def _header(self, entry: dict) -> dict:
        if entry["file"] not in self._headers:
            with (self.path / entry["file"]).open(encoding="utf-8") as f:
                self._headers[entry["file"]] = json.loads(f.readline())["collection"]
        return self._headers[entry["file"]]

    def _records(self, entry: dict, offset: int, limit: int) -> list[dict]:
        f, pos = self._readers.get(entry["file"], (None, -1))
        if f is None or pos != offset:
            if f is not None:
                f.close()
            f = (self.path / entry["file"]).open(encoding="utf-8")
            f.readline()
            for _ in islice(f, offset):
                pass
        lines = list(islice(f, limit))
        self._readers[entry["file"]] = (f, offset + len(lines))
        return [json.loads(line) for line in lines]

    async def resolve_tenant(self) -> str:
        self._manifest()
        assert self.tenant is not None
        return self.tenant

    async def databases(self) -> list[str]:
        return sorted({db for db, _ in self._manifest()})

    async def collections(self, database: str) -> list[dict]:
        entries = [e for (db, _), e in self._manifest().items() if db == database]
        if not entries:
            raise SourceError(404, f"database [{database}] is not in the export")
        return [self._header(e) for e in entries]

    async def count(self, database: str, collection_id: str) -> int:
        return int(self._entry(database, collection_id)["records"])

    async def get_page(self, database: str, collection_id: str, offset: int, limit: int) -> dict:
        records = self._records(self._entry(database, collection_id), offset, limit)
        page: dict[str, Any] = {"ids": [r["id"] for r in records]}
        for key, plural in (*FIELDS, ("uri", "uris")):
            page[plural] = [r.get(key) for r in records]
        return page

    async def query(
        self, database: str, collection_id: str, embeddings: list, n_results: int
    ) -> dict:
        """Exact nearest neighbours, scanning the file."""
        entry = self._entry(database, collection_id)
        coll = self._header(entry)
        space = normalize_configuration(coll.get("configuration_json"), coll.get("metadata"))[
            "hnsw"
        ]["space"]
        queries = np.asarray(embeddings, dtype=np.float32)
        best_ids = [np.array([], dtype=object) for _ in queries]
        best = [np.array([]) for _ in queries]
        offset = 0
        while records := self._records(entry, offset, 4096):
            offset += len(records)
            ids = np.array([r["id"] for r in records], dtype=object)
            dist = _distances(space, queries, np.asarray([r["embedding"] for r in records]))
            for i in range(len(queries)):
                d = np.concatenate([best[i], dist[i]])
                keep = np.argsort(d, kind="stable")[:n_results]
                best[i], best_ids[i] = d[keep], np.concatenate([best_ids[i], ids])[keep]
        return {
            "ids": [list(ids) for ids in best_ids],
            "distances": [list(map(float, d)) for d in best],
        }
