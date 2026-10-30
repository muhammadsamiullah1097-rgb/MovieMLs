from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dataset_summary():
    response = client.get("/api/summary")
    assert response.status_code == 200
    assert response.json()["movie_rows"] > 1000
    assert response.json()["rating_rows"] > 1000


def test_model_lists():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["classification"]) == 12
    assert len(data["regression"]) == 8
    assert len(data["text"]) == 2
