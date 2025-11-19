from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_generate_endpoint():
    response = client.post("/api/generate", json={
        "subject": "Test Subject",
        "body": "Test Body",
        "mode": "lora"
    })
    assert response.status_code == 200
    data = response.json()
    assert "lora_reply" in data
    assert data["used_mode"] == "lora"
