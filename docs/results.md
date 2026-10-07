# Measured results

These results come from the downloaded workbook and a complete local pipeline run, not illustrative numbers.

- Model run: `20261007T030225997489Z`
- Source workbook identity: `61b1d77dd6d9298f`
- Trained products: 55
- Products passing product-level evidence checks: 34
- Simulation week: 2012-01-11

## Forecast evaluation

| Period | Model | MAE (units) | WAPE | Mean bias (units) | Common rows |
|---|---|---:|---:|---:|---:|
| validation | baseline | 9.1542 | 46.64% | -0.0188 | 90,812 |
| validation | log_log | 6.6724 | 34.00% | -0.1539 | 90,812 |
| validation | xgboost | 8.7146 | 44.40% | +0.4691 | 90,812 |
| test | baseline | 7.9240 | 43.37% | +0.1049 | 85,742 |
| test | log_log | 7.3956 | 40.48% | +0.1193 | 85,742 |
| test | xgboost | 8.6346 | 47.26% | +0.4463 | 85,742 |

The interpretable model wins on this test period. This does not establish a general advantage over boosted trees; the challenger uses a modest fixed specification. Its weaker result is retained rather than hidden. Validation excluded 459 rows and testing excluded 748 because a common baseline/statistical prediction was unavailable. Product 3500068914 has no test-period observations.

The statistical model improves test MAE by approximately 6.7% relative to the rolling baseline. That is a predictive error reduction, not a revenue improvement. The baseline is updated one week at a time from past actual sales.

## Reproducible product walkthrough

Select **ORAL HYGIENE PRODUCTS → PL BL MINT ANTSPTC RINSE → HYDE PARK · 24991** in the dashboard.

- Product UPC: `1111035398`
- Elasticity: -0.599; approximate 95% confidence interval -0.770 to -0.428
- Latest baseline: $2.95, no promotional support
- Best tested candidate: $3.25, display support
- Cost assumptions: absent; promotion cost defaults to zero, so no profit conclusion is available

| Simulated measure | Baseline | Proposed |
|---|---:|---:|
| Units | 9.78 | 11.05 |
| Revenue | $28.85 | $35.92 |
| Conditional mean units interval | 8.98–10.65 | 9.10–13.42 |

Estimated revenue changes by 24.49% in this combined price/display scenario. This is an observational simulation, not a realized gain. The estimate depends on the assumed display effect and ignores display cost unless entered. The price is above the current reference price, so its displayed discount is negative. It is near the upper candidate boundary, which is expected with an estimated elasticity between -1 and 0.

The larger predicted quantity despite a higher price comes from the positive display coefficient. This is a useful point to challenge in an interview: additional support is not free in practice, the combination may be sparsely observed, and model associations need experimental validation.

Full inputs, intervals and outputs are in `reports/demo_result.json`. `uv run --no-editable retail-pricing demo` regenerates a recorded example against the active model, and `scripts/verify_local.py` checks API reproduction. Scenario IDs change on each run because each saved comparison is a new record.

## Verification performed

- Complete fetch/ingest/load/train/demo workflow against native PostgreSQL.
- Unit, synthetic-model, API, persistence and Streamlit interaction tests.
- Three notebooks executed from top to bottom.
- Live API scenario save/retrieve/reproduction checks.
- Browser inspection of the local dashboard.
- Docker Compose configuration validation. Container startup was not exercised because the Docker engine was unavailable; the native services were exercised instead.
