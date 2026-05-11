from app.main import app
from fastapi.testclient import TestClient


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_schema() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need an assessment."}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert {"reply", "recommendations", "end_of_conversation"} == set(body)
        assert isinstance(body["recommendations"], list)
