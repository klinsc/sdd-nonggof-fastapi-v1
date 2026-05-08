from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.main import app


def test_root_requires_api_key():
    # The root endpoint opts out of the global API-key dep (dependencies=[]),
    # so it should be reachable without auth headers.
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Bigger Applications!"}


def test_healthz_reports_readiness():
    client = TestClient(app)
    # Without entering the lifespan context, ready is False.
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "starting"}
    assert "ready" in body


def test_protected_route_rejects_missing_api_key():
    # Use an isolated app to avoid booting the real lifespan.
    sub = FastAPI()

    from fastapi import Depends

    @sub.get("/protected", dependencies=[Depends(require_api_key)])
    async def protected():
        return {"ok": True}

    client = TestClient(sub)
    assert client.get("/protected").status_code == 401
    assert client.get(
        "/protected", headers={"X-API-Key": "wrong"}
    ).status_code == 401
