from fastapi.testclient import TestClient

from main import app, cosine_score

client = TestClient(app)


def test_cosine_score_prefers_aligned_vectors() -> None:
    assert cosine_score({"chess": 1.0}, {"chess": 1.0}) == 1.0
    assert cosine_score({"chess": 1.0}, {"music": 1.0}) == 0.0


def test_recommendation_is_ranked_and_deterministic() -> None:
    response = client.post(
        "/v1/recommend",
        headers={"X-Request-Id": "req-1"},
        json={
            "profile": {"chess": 1.0, "ai": 0.5},
            "items": [
                {"id": "b", "features": {"music": 1.0}},
                {"id": "a", "features": {"chess": 1.0}},
                {"id": "c", "features": {"chess": 1.0, "ai": 0.5}},
            ],
            "limit": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-1"
    assert [item["id"] for item in payload["results"]] == ["c", "a"]


def test_health_endpoints() -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
