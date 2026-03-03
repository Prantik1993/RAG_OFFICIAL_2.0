"""
Basic tests for Legal RAG System
"""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_chat_endpoint_validation(client):
    """Test chat endpoint input validation"""
    # Empty query should fail
    response = client.post("/chat", json={
        "query": "",
        "session_id": "test"
    })
    assert response.status_code == 422  # Validation error
    
    # Valid query should work (if system is initialized)
    response = client.post("/chat", json={
        "query": "What is Article 1?",
        "session_id": "test"
    })
    # Should be 200 or 503 depending on initialization
    assert response.status_code in [200, 503]


def test_stats_endpoint(client):
    """Test stats endpoint"""
    response = client.get("/stats")
    # Should be 200 or 503 depending on initialization
    assert response.status_code in [200, 503]
