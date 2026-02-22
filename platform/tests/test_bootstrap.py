# platform/tests/test_bootstrap.py
import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "x" * 64)

import asyncio
from unittest.mock import patch, MagicMock

from api.models.run import Base
from api.deps import engine


async def _create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
loop.run_until_complete(_create_tables())

from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)


def test_create_bootstrap_returns_201():
    with patch("api.routes.bootstrap.get_similarity_model", return_value=None),          patch("api.routes.bootstrap.get_all_modalities", return_value=[]):
        response = client.post("/api/v1/bootstrap", json={
            "name": "THz TDS",
            "description": "Terahertz time-domain spectroscopy with pulsed THz source and time-gated detection system",
            "physics_class": "coherent",
        })
    assert response.status_code == 201, response.text
    data = response.json()
    assert "id" in data
    assert data["status"] == "draft"
    assert data["name"] == "THz TDS"


def test_bootstrap_short_description_rejected():
    response = client.post("/api/v1/bootstrap", json={
        "name": "THz",
        "description": "too short",
    })
    assert response.status_code == 422


def test_bootstrap_has_viability_checklist():
    with patch("api.routes.bootstrap.get_similarity_model", return_value=None),          patch("api.routes.bootstrap.get_all_modalities", return_value=[]):
        response = client.post("/api/v1/bootstrap", json={
            "name": "X-ray Phase Contrast",
            "description": "X-ray phase contrast imaging using Talbot-Lau interferometry for soft tissue contrast",
            "physics_class": "incoherent",
        })
    assert response.status_code == 201
    data = response.json()
    assert len(data["viability_checklist"]) >= 6


def test_bootstrap_get_not_found():
    response = client.get("/api/v1/bootstrap/does-not-exist")
    assert response.status_code == 404
