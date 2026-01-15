import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture
def client():
    """
    Creates a TestClient instance wrapped in a context manager.
    This ensures the 'startup' event (loading the DB) runs before tests start.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Does the API start up correctly?"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_chat_response(client):
    """Does the chat endpoint return a valid answer?"""
    payload = {
        "query": "What are the conditions for consent?",
        "session_id": "test_session_123"
    }
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "sources" in data

def test_empty_input_guardrail(client):
    """Does the Guardrail stop empty inputs?"""
    payload = {"query": "", "session_id": "test_fail"}
    response = client.post("/chat", json=payload)
    
    # Should be 400 Bad Request, NOT 500
    assert response.status_code == 400
    assert "detail" in response.json()