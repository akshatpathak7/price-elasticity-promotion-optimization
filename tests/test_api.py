import pytest
from fastapi.testclient import TestClient

from retail_pricing.api import app, get_service
from retail_pricing.schemas import ScenarioRequest


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_catalog_and_health(client):
    assert client.get("/health").status_code == 200
    assert client.get("/products").json()["products"][0]["product_id"] == "p1"
    assert client.get("/products/p1/analysis?store_id=1").status_code == 200
    assert client.get("/models/metrics").status_code == 200


def test_api_matches_shared_engine_and_saved_scenario(client, service):
    payload = {"product_id": "p1", "store_id": "1", "price": 2.4}
    response = client.post("/scenarios", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    expected = service.simulate(ScenarioRequest(**payload), persist=False)
    assert result["proposed"] == expected["proposed"]
    saved = client.get("/scenarios/" + result["scenario_id"]).json()
    assert saved == result


def test_http_errors(client):
    assert client.get("/products/unknown/analysis").status_code == 404
    assert (
        client.post(
            "/scenarios", json={"product_id": "p1", "store_id": "1", "price": -1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/scenarios", json={"product_id": "p1", "store_id": "1", "price": 999}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/recommendations", json={"product_id": "p1", "store_id": "1", "min_margin_pct": 10}
        ).status_code
        == 422
    )


def test_recommendation_is_json_serializable(client):
    r = client.post("/recommendations", json={"product_id": "p1", "store_id": "1"})
    assert r.status_code == 200, r.text
    assert r.json()["recommendation"] is not None
