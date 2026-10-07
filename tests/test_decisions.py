import pytest
from pydantic import ValidationError

from retail_pricing.schemas import RecommendationRequest, ScenarioRequest
from retail_pricing.service import outcome


def test_revenue_and_margin_arithmetic():
    result = outcome(4, 100, 90, 110, unit_cost=2, promotion_cost=20)
    assert result["revenue"] == 400
    assert result["contribution_profit"] == 180
    assert result["margin_pct"] == 45
    assert result["revenue_interval"] == [360, 440]
    assert outcome(4, 100, 90, 110)["contribution_profit"] is None


def test_price_and_discount_are_exclusive():
    with pytest.raises(ValidationError):
        ScenarioRequest(product_id="p1", store_id="1", price=2, discount_pct=10)


def test_margin_requires_cost():
    with pytest.raises(ValidationError):
        RecommendationRequest(product_id="p1", store_id="1", min_margin_pct=20)


def test_baseline_has_no_change(service):
    result = service.simulate(ScenarioRequest(product_id="p1", store_id="1"), persist=False)
    assert result["changes"]["revenue_pct"] == pytest.approx(0)
    assert result["baseline"]["units"] == pytest.approx(result["proposed"]["units"])


def test_discount_resolution(service):
    result = service.simulate(
        ScenarioRequest(
            product_id="p1",
            store_id="1",
            reference_price=3,
            discount_pct=20,
            support={"display": 0},
        ),
        persist=False,
    )
    assert result["proposed"]["price"] == 2.4


def test_outside_price_support_rejected(service):
    with pytest.raises(ValueError, match="outside"):
        service.simulate(ScenarioRequest(product_id="p1", store_id="1", price=999), persist=False)


def test_unsupported_product(service):
    with pytest.raises(LookupError):
        service.simulate(ScenarioRequest(product_id="missing", store_id="1"), persist=False)


def test_unsupported_combination(service):
    with pytest.raises(ValueError, match="combination"):
        service.simulate(
            ScenarioRequest(product_id="p1", store_id="1", support={"feature": 1}), persist=False
        )


def test_candidates_rank_and_baseline_inclusion(service):
    r = service.recommend(RecommendationRequest(product_id="p1", store_id="1", max_discount_pct=90))
    assert r["status"] == "recommended"
    assert r["recommendation"]["revenue"] == max(c["revenue"] for c in r["candidates"])
    assert any(c["unchanged_baseline"] for c in r["candidates"])
    c = r["recommendation"]
    check = service.simulate(
        ScenarioRequest(product_id="p1", store_id="1", price=c["price"], support=c["support"]),
        persist=False,
    )
    assert c["units"] == pytest.approx(check["proposed"]["units"])


def test_margin_infeasible(service):
    r = service.recommend(
        RecommendationRequest(product_id="p1", store_id="1", unit_cost=100, min_margin_pct=20)
    )
    assert r["status"] == "infeasible" and r["recommendation"] is None


def test_eligibility_is_not_silently_relaxed(service):
    service.summary["diagnostics"]["p1"]["ineligibility_reasons"] = ["Uncertain elasticity."]
    r = service.recommend(RecommendationRequest(product_id="p1", store_id="1"))
    assert r["status"] == "insufficient_evidence"
    assert r["recommendation"] is None
    assert r["candidates"]
