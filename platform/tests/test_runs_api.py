import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "x" * 64)

import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.models.run import Base
from api.deps import engine


# Create tables synchronously before tests run
async def _create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
loop.run_until_complete(_create_tables())

from api.main import app
client = TestClient(app)


def test_create_run_returns_201():
    with patch("api.routes.runs.dispatch_pwm_run") as mock_task:
        mock_task.delay.return_value = MagicMock(id="celery-task-123")
        response = client.post("/api/v1/runs", json={
            "prompt": "Simulate CT scan",
            "compute_mode": "cpu",
        })
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] == "queued"


def test_create_run_no_prompt_or_spec_returns_400():
    response = client.post("/api/v1/runs", json={"compute_mode": "cpu"})
    assert response.status_code == 400


def test_list_runs_returns_list():
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_run_not_found():
    response = client.get("/api/v1/runs/does-not-exist")
    assert response.status_code == 404
