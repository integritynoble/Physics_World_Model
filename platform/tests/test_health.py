import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "x" * 64)

from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)

def test_health_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_json_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"

def test_health_has_version():
    response = client.get("/health")
    assert "version" in response.json()
