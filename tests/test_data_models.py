import numpy as np
import pandas as pd
import pytest

from retail_pricing.data import validate
from retail_pricing.modeling import (
    design,
    eligibility,
    fit_elasticity,
    metrics,
    predict_elasticity,
    rolling_baseline,
    split_calendar,
)


def test_duplicate_keys_fail(sales):
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([sales, sales.iloc[:1]]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("price", 0),
        ("price", np.inf),
        ("base_price", -1),
        ("units", -1),
        ("display", 2),
        ("revenue", np.nan),
    ],
)
def test_invalid_values_quarantined(sales, column, value):
    sample = sales.copy().astype({column: float})
    sample.loc[0, column] = value
    clean, bad, report = validate(sample)
    assert len(bad) == 1 and len(clean) == len(sample) - 1
    assert report["rejected_rows"] == 1


def test_missing_week_is_not_zero_filled(sales):
    clean, _, report = validate(sales.drop(index=20))
    assert len(clean) == len(sales) - 1
    assert report["missing_pair_weeks"] == 1


def test_zero_sales_retained_but_excluded_from_log_fit(sales):
    sample = sales.copy()
    sample.loc[0, ["units", "revenue"]] = 0
    clean, _, report = validate(sample)
    assert report["zero_sales_retained"] == 1
    fitted = fit_elasticity(clean, clean.week.min())
    assert fitted["diagnostics"]["zero_sales_excluded"] == 1


def test_calendar_split(sales):
    parts = split_calendar(sales)
    assert [p.week.nunique() for p in parts.values()] == [104, 26, 26]
    assert parts["train"].week.max() < parts["validation"].week.min()
    assert parts["validation"].week.max() < parts["test"].week.min()


def test_missing_entire_week_does_not_shift_split(sales):
    removed = sales[sales.week != sorted(sales.week.unique())[10]]
    assert split_calendar(removed)["train"].week.max() == split_calendar(sales)["train"].week.max()


def test_baseline_never_reads_current_or_future_units(sales):
    original = rolling_baseline(sales)
    altered = sales.copy()
    altered.loc[altered.week >= sorted(sales.week.unique())[140], "units"] = 999999
    changed = rolling_baseline(altered)
    mask = sales.week <= sorted(sales.week.unique())[140]
    np.testing.assert_allclose(original[mask], changed[mask], equal_nan=True)


def test_known_elasticity_recovered(fitted):
    assert fitted["diagnostics"]["elasticity"] == pytest.approx(-1.6, abs=0.06)
    assert fitted["smearing"] >= 1


def test_single_row_predictions_keep_store_effects(fitted, sales):
    batch = sales.iloc[[0, 156, 312, 468]]
    grouped = predict_elasticity(fitted, batch)
    individual = np.array([predict_elasticity(fitted, batch.iloc[[i]])[0] for i in range(4)])
    np.testing.assert_allclose(grouped, individual)


def test_forecast_price_ratio_matches_elasticity(fitted, sales):
    one = sales.iloc[[0]].copy()
    two = one.copy()
    two["price"] *= 1.1
    q1 = predict_elasticity(fitted, one)[0]
    q2 = predict_elasticity(fitted, two)[0]
    assert q2 / q1 == pytest.approx(1.1 ** fitted["diagnostics"]["elasticity"])


def test_training_design_ignores_target_leaking_columns(sales):
    matrix = design(sales, sales.week.min())
    assert not {"units", "revenue", "visits", "hhs", "base_price"} & set(matrix)


def test_future_mutation_does_not_change_fit(sales, fitted):
    modified = sales.copy()
    modified.loc[modified.week > sorted(sales.week.unique())[103], ["units", "price"]] = 99999
    train = split_calendar(modified)["train"]
    again = fit_elasticity(train, sales.week.min())
    np.testing.assert_allclose(fitted["coefficients"], again["coefficients"])
    assert fitted["smearing"] == again["smearing"]


def test_uncertain_elasticity_blocks_recommendation(fitted):
    diag = {**fitted["diagnostics"], "elasticity_ci": [-2, 0.2]}
    reasons = eligibility(diag, {"log_log": {"mae": 1}, "baseline": {"mae": 2}})
    assert any("confidently negative" in r for r in reasons)


def test_zero_denominator_metric():
    assert metrics([0, 0], [1, 1])["wape"] is None
