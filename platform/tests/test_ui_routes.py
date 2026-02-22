import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "x" * 64)

from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)


def test_home_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "PWM" in r.text


def test_runs_page_returns_html():
    r = client.get("/runs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_bootstrap_wizard_returns_html():
    r = client.get("/bootstrap/new")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_datasets_page_returns_html():
    r = client.get("/datasets")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_modalities_page_returns_html():
    r = client.get("/modalities")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
