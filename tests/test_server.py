import base64
import uuid

import httpx
import numpy as np
import pytest

from pgchroma.server import Settings, create_app

BASE = "/api/v2/tenants/default_tenant/databases/default_database"


@pytest.fixture
async def make_api(database_url):
    apps = []

    async def make(**settings):
        schema = f"t_{uuid.uuid4().hex[:12]}"
        app = create_app(Settings(database_url=database_url, schema=schema, **settings))
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        apps.append((app, ctx, schema))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    yield make
    for app, ctx, schema in apps:
        async with app.state.store.pool.connection() as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await ctx.__aexit__(None, None, None)


@pytest.fixture
async def api(make_api):
    return await make_api()


async def test_heartbeat_version_preflight_identity(api):
    hb = (await api.get("/api/v2/heartbeat")).json()
    assert isinstance(hb["nanosecond heartbeat"], int)
    assert (await api.get("/api/v2/version")).json() == "1.0.0"
    assert (await api.get("/api/v2/pre-flight-checks")).json() == {
        "max_batch_size": 5461,
        "supports_base64_encoding": True,
    }
    assert (await api.get("/api/v2/auth/identity")).json() == {
        "user_id": "",
        "tenant": "default_tenant",
        "databases": ["default_database"],
    }


async def test_records_flow_with_base64(api):
    r = await api.post(
        f"{BASE}/collections", json={"name": "docs", "metadata": None, "get_or_create": False}
    )
    assert r.status_code == 200
    cid = r.json()["id"]
    packed = [
        base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()
        for v in ([1, 0, 0], [0, 1, 0])
    ]
    r = await api.post(
        f"{BASE}/collections/{cid}/add",
        json={
            "ids": ["a", "b"],
            "embeddings": packed,
            "metadatas": [{"k": 1}, None],
            "documents": ["A", "B"],
            "uris": None,
        },
    )
    assert (r.status_code, r.json()) == (201, {})
    r = await api.post(
        f"{BASE}/collections/{cid}/query",
        json={
            "query_embeddings": [[1.0, 0.1, 0.0]],
            "n_results": 1,
            "include": ["distances", "documents", "embeddings"],
        },
    )
    body = r.json()
    assert body["ids"] == [["a"]] and body["documents"] == [["A"]]
    assert body["embeddings"] == [[[1.0, 0.0, 0.0]]]
    assert abs(body["distances"][0][0] - 0.01) < 1e-6
    assert (await api.get(f"{BASE}/collections/{cid}/count")).json() == 2
    r = await api.post(f"{BASE}/collections/{cid}/delete", json={"ids": ["a"]})
    assert r.json() == {"deleted": 1}


async def test_not_found_body(api):
    r = await api.get(f"{BASE}/collections/nope")
    assert r.status_code == 404
    assert r.json() == {"error": "NotFoundError", "message": "Collection [nope] does not exist"}


async def test_get_collection_by_name_or_id_and_crn(api):
    cid = (await api.post(f"{BASE}/collections", json={"name": "docs"})).json()["id"]
    assert (await api.get(f"{BASE}/collections/docs")).json()["id"] == cid
    assert (await api.get(f"{BASE}/collections/by-id/{cid}")).json()["name"] == "docs"
    assert (await api.get(f"{BASE}/collections/{cid}")).json()["name"] == "docs"
    assert (await api.get("/api/v2/collections/default_tenant:default_database:docs")).json()[
        "id"
    ] == cid


async def test_invalid_json_is_400(api):
    r = await api.post(
        f"{BASE}/collections", content=b"{nope", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400 and r.json()["error"] == "InvalidArgumentError"


async def test_auth_tokens(make_api):
    api = await make_api(auth_tokens=["s3cret"])
    assert (await api.get("/api/v2/heartbeat")).status_code == 200
    r = await api.get(f"{BASE}/collections")
    assert r.status_code == 401 and r.json()["error"] == "AuthError"
    assert (
        await api.get(f"{BASE}/collections", headers={"x-chroma-token": "s3cret"})
    ).status_code == 200
    assert (
        await api.get(f"{BASE}/collections", headers={"Authorization": "Bearer s3cret"})
    ).status_code == 200
    assert (
        await api.get(f"{BASE}/collections", headers={"x-chroma-token": "wrong"})
    ).status_code == 401


async def test_reset_disabled_by_default_and_enabled_on_request(make_api):
    api = await make_api()
    assert (await api.post("/api/v2/reset")).status_code == 401
    api = await make_api(allow_reset=True)
    await api.post(f"{BASE}/collections", json={"name": "docs"})
    r = await api.post("/api/v2/reset")
    assert (r.status_code, r.json()) == (200, True)
    assert (await api.get(f"{BASE}/collections_count")).json() == 0


async def test_unsupported_and_v1(api):
    cid = (await api.post(f"{BASE}/collections", json={"name": "docs"})).json()["id"]
    r = await api.post(f"{BASE}/collections/{cid}/functions/attach", json={})
    assert r.status_code == 501 and r.json()["error"] == "ChromaError"
    r = await api.get("/api/v1/heartbeat")
    assert r.status_code == 410 and "v1" in r.json()["message"]
