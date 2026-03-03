"""
API smoke tests — uses TestClient (no running server needed).
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "name" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_chat_empty_query(client):
    r = client.post("/chat", json={"query": "", "session_id": "test"})
    assert r.status_code == 422


def test_chat_valid_query(client):
    r = client.post("/chat", json={"query": "What is Article 1?", "session_id": "test"})
    # 200 if engine ready, 503 if not (CI without API key)
    assert r.status_code in (200, 503)
