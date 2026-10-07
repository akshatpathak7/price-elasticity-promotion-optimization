"""Verify the real PostgreSQL-backed application after setup and API startup."""

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
root = Path(__file__).resolve().parents[1]
demo = json.loads((root / "reports/demo_result.json").read_text())
request = demo["scenario"]["assumptions"]
payload = {
    k: request[k]
    for k in [
        "product_id",
        "store_id",
        "price",
        "reference_price",
        "discount_pct",
        "support",
        "unit_cost",
        "promotion_cost",
        "baseline_promotion_cost",
    ]
}
with httpx.Client(base_url=os.getenv("API_URL", "http://localhost:8000"), timeout=30) as client:
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/products").json()["products"]) > 0
    r = client.post("/scenarios", json=payload)
    r.raise_for_status()
    result = r.json()
    retrieved = client.get("/scenarios/" + result["scenario_id"])
    retrieved.raise_for_status()
    assert retrieved.json() == result
    assert abs(result["proposed"]["revenue"] - demo["scenario"]["proposed"]["revenue"]) < 1e-8
    recommendation = client.post("/recommendations", json=demo["recommendation"]["assumptions"])
    recommendation.raise_for_status()
    assert recommendation.json()["status"] == "recommended"
    assert client.post("/scenarios", json={**payload, "price": -1}).status_code == 422
    assert client.get("/products/unknown/analysis").status_code == 404
print(
    "PASS: PostgreSQL-backed health, catalog, scenario persistence, reproduction, recommendation and errors."
)
