"""Collection names, configuration and schema, normalized the way Chroma 1.5 returns them."""

from __future__ import annotations

import copy
import re
from typing import Any

from .errors import InvalidArgumentError

SPACES = ("l2", "cosine", "ip")

HNSW_DEFAULTS: dict[str, Any] = {
    "ef_construction": 100,
    "ef_search": 100,
    "max_neighbors": 16,
    "resize_factor": 1.2,
    "sync_threshold": 1000,
}

# Pre-1.0 Chroma stored index settings in collection metadata.
_LEGACY_METADATA_KEYS = {
    "hnsw:construction_ef": "ef_construction",
    "hnsw:search_ef": "ef_search",
    "hnsw:M": "max_neighbors",
    "hnsw:resize_factor": "resize_factor",
    "hnsw:sync_threshold": "sync_threshold",
}

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def validate_collection_name(name: Any) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise InvalidArgumentError(
            "Validation error: name: Expected a name containing 3-512 characters from "
            "[a-zA-Z0-9._-], starting and ending with a character in [a-zA-Z0-9]. "
            f"Got: {name}"
        )
    return name


def normalize_configuration(configuration: dict | None, metadata: dict | None) -> dict:
    cfg = configuration or {}
    if not isinstance(cfg, dict):
        raise InvalidArgumentError(f"Expected configuration to be an object, got {cfg!r}")
    hnsw_in = cfg.get("hnsw") or {}
    spann_in = cfg.get("spann") or {}
    meta = metadata or {}
    space = hnsw_in.get("space") or spann_in.get("space") or meta.get("hnsw:space") or "l2"
    if space not in SPACES:
        raise InvalidArgumentError(f"Invalid space {space!r}: expected one of {', '.join(SPACES)}")
    hnsw: dict[str, Any] = {"space": space}
    for key, default in HNSW_DEFAULTS.items():
        legacy = next(
            (meta[k] for k, v in _LEGACY_METADATA_KEYS.items() if v == key and k in meta), None
        )
        hnsw[key] = hnsw_in.get(key, legacy if legacy is not None else default)
    return {"hnsw": hnsw, "spann": None, "embedding_function": cfg.get("embedding_function")}


def merge_configuration(stored: dict, new: dict | None) -> dict:
    if not new:
        return stored
    merged = copy.deepcopy(stored)
    hnsw_new = new.get("hnsw") or {}
    if "space" in hnsw_new and hnsw_new["space"] != merged["hnsw"]["space"]:
        raise InvalidArgumentError("The distance space of an existing collection cannot be changed")
    merged["hnsw"].update({k: v for k, v in hnsw_new.items() if v is not None})
    if new.get("embedding_function") is not None:
        merged["embedding_function"] = new["embedding_function"]
    return merged


def _schema_hnsw(hnsw: dict) -> dict:
    return {
        "ef_construction": hnsw["ef_construction"],
        "max_neighbors": hnsw["max_neighbors"],
        "ef_search": hnsw["ef_search"],
        "num_threads": 16,
        "batch_size": 100,
        "sync_threshold": hnsw["sync_threshold"],
        "resize_factor": hnsw["resize_factor"],
    }


def default_schema(configuration: dict) -> dict:
    hnsw = configuration["hnsw"]
    ef = configuration.get("embedding_function")
    vector_config: dict[str, Any] = {"space": hnsw["space"]}
    if ef is not None:
        vector_config["embedding_function"] = ef
    vector_config["hnsw"] = _schema_hnsw(hnsw)
    embedding_config = {k: v for k, v in vector_config.items() if k != "hnsw"}
    embedding_config["source_key"] = "#document"
    embedding_config["hnsw"] = _schema_hnsw(hnsw)
    return {
        "defaults": {
            "string": {
                "fts_index": {"enabled": False, "config": {}},
                "string_inverted_index": {"enabled": True, "config": {}},
            },
            "float_list": {"vector_index": {"enabled": False, "config": vector_config}},
            "sparse_vector": {
                "sparse_vector_index": {
                    "enabled": False,
                    "config": {"embedding_function": {"type": "unknown"}, "bm25": False},
                }
            },
            "int": {"int_inverted_index": {"enabled": True, "config": {}}},
            "float": {"float_inverted_index": {"enabled": True, "config": {}}},
            "bool": {"bool_inverted_index": {"enabled": True, "config": {}}},
        },
        "keys": {
            "#document": {
                "string": {
                    "fts_index": {"enabled": True, "config": {}},
                    "string_inverted_index": {"enabled": False, "config": {}},
                }
            },
            "#embedding": {
                "float_list": {"vector_index": {"enabled": True, "config": embedding_config}}
            },
        },
    }
