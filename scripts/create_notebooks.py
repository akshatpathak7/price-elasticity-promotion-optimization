"""Rebuild the three small, executable analysis notebooks from their learning outline."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOT = """from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display
ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
from retail_pricing.config import DATA, ARTIFACTS, REPORTS
observations = pd.read_parquet(DATA / "processed/observations.parquet")
products = pd.read_parquet(DATA / "processed/products.parquet")
quality = json.loads((REPORTS / "data_quality.json").read_text())
"""


def write(name, cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {
        "display_name": "Retail Pricing",
        "language": "python",
        "name": "retail-pricing",
    }
    book.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        for kind, body in cells
    ]
    nbf.write(book, ROOT / "notebooks" / name)


write(
    "01_data_exploration.ipynb",
    [
        (
            "md",
            "# 01 · Understand the sales data\n\nStart with the grain and data quality before modeling. Source: [dunnhumby Breakfast at the Frat](https://www.dunnhumby.com/source-files/), a representation of historical retail data. Run the setup command in the README first.",
        ),
        ("code", BOOT),
        (
            "md",
            "## Coverage and rejected observations\nMissing weeks are not zero sales. Lookup counts can differ from observed counts. The source guide and workbook are preserved under `data/raw`.",
        ),
        (
            "code",
            'display(pd.Series(quality).to_frame("value"))\nrejected = pd.read_parquet(DATA / "processed/rejected.parquet")\ndisplay(rejected[["price", "base_price", "units", "reason"]].head(10))',
        ),
        (
            "md",
            "## Which products dominate revenue?\nUse source spend for historical revenue. The simulator later calculates price × predicted units.",
        ),
        (
            "code",
            """ranking = observations.groupby("product_id").agg(units=("units", "sum"), revenue=("revenue", "sum"), rows=("units", "size")).reset_index().merge(products, on="product_id")
display(ranking.sort_values("revenue", ascending=False)[["description", "category", "units", "revenue", "rows"]].head(10))""",
        ),
        (
            "code",
            """weekly = observations.groupby("week", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
px.line(weekly, x="week", y="units", title="Observed weekly units across the dataset").show(renderer="plotly_mimetype")""",
        ),
        (
            "md",
            "## Promotion coverage and unusual purchases\nDo not silently trim unusually large purchases. Units per visit can identify observations worth inspecting; without inventory data we cannot label low sales as stockouts.",
        ),
        (
            "code",
            """display(observations.groupby(["feature", "display", "tpr_only"]).size().rename("rows").reset_index())
ratios = observations.assign(units_per_visit=observations.units / observations.visits, visits_per_household=observations.visits / observations.hhs)
display(ratios.nlargest(10, "units_per_visit")[["product_id", "store_id", "week", "units", "visits", "units_per_visit"]])""",
        ),
        (
            "md",
            "## Takeaway\nThe data contains useful price and promotion variation but not randomized interventions or inventory. Keep source anomalies explicit, and split on time before fitting transformations. See `docs/data.md` for the full cleaning policy.",
        ),
    ],
)

write(
    "02_model_comparison.ipynb",
    [
        (
            "md",
            "# 02 · Estimate elasticity and evaluate predictions\n\nThe statistical model answers an interpretable question; XGBoost is a forecasting challenger. Neither forecast accuracy nor a negative price coefficient establishes causality.",
        ),
        (
            "code",
            BOOT
            + '\nfrom retail_pricing.modeling import split_calendar, fit_elasticity, predict_elasticity, load_artifacts\nartifacts, summary = load_artifacts()\nparts = split_calendar(observations)\ndisplay(pd.DataFrame([{ "split": name, "start": frame.week.min(), "end": frame.week.max(), "rows": len(frame)} for name, frame in parts.items()]))',
        ),
        (
            "md",
            "## Refit one product to inspect the estimator\nThis cell fits only training rows. Store intercepts, support flags and calendar effects adjust observed associations. The full pipeline uses the identical function for every product.",
        ),
        (
            "code",
            """product_id = next(p for p, d in summary["diagnostics"].items() if not d["ineligibility_reasons"])
sample = parts["train"].query("product_id == @product_id")
model = fit_elasticity(sample, observations.week.min())
display(pd.Series(model["diagnostics"]).to_frame("value"))
print("Training smearing factor:", model["smearing"])""",
        ),
        (
            "md",
            "## Compare all models on common rows\nMAE is in units. WAPE is scale-independent at the aggregate level. Positive bias means overprediction. The baseline updates from prior observed weeks, never current-week sales.",
        ),
        (
            "code",
            """scores = [{"split": split, "model": name, **score} for split in ["validation", "test"] for name, score in summary["metrics"][split]["overall"].items()]
display(pd.DataFrame(scores))
px.bar(pd.DataFrame(scores), x="model", y="wape", color="split", barmode="group", title="Held-out forecasting error").show(renderer="plotly_mimetype")""",
        ),
        (
            "md",
            "## Inspect errors instead of only one headline score\nModel ranking may vary by product, and aggregate performance can conceal weak products. The test period is reported after model choices are fixed.",
        ),
        (
            "code",
            """predictions = pd.read_parquet(ARTIFACTS / summary["version"] / "predictions.parquet")
selected = predictions.query("product_id == @product_id and split == 'test'")
weekly = selected.groupby("week", as_index=False)[["units", "baseline", "log_log", "xgboost"]].sum()
px.line(weekly, x="week", y=["units", "baseline", "log_log", "xgboost"], title="Selected product: test-period units across stores").show(renderer="plotly_mimetype")
display(pd.DataFrame(summary["metrics"]["test"]["by_product"][product_id]).T)""",
        ),
        (
            "code",
            """diagnostics = pd.DataFrame(summary["diagnostics"]).T
print("Modeled products:", len(diagnostics))
print("Products passing evidence checks:", diagnostics.ineligibility_reasons.map(len).eq(0).sum())
display(diagnostics[["elasticity", "elasticity_ci", "ineligibility_reasons"]].head(12))""",
        ),
        (
            "md",
            "## Interpretation\nUse measured results even when the simpler model wins. Eligibility is set using training diagnostics and validation performance, never test performance. The fitted model stays frozen for the demonstration; refitting would require a separately versioned evaluation. See `docs/methodology.md`.",
        ),
    ],
)

write(
    "03_pricing_decisions.ipynb",
    [
        (
            "md",
            "# 03 · Turn an estimate into a bounded decision\n\nReproduce the saved demonstration with the shared decision engine. PostgreSQL must be running. All revenue changes below are simulated, not realized business improvements.",
        ),
        (
            "code",
            BOOT
            + """
from retail_pricing import db
from retail_pricing.service import PricingService
from retail_pricing.schemas import ScenarioRequest, RecommendationRequest
service = PricingService(db.engine())
demo = json.loads((REPORTS / "demo_result.json").read_text())
product_id = demo["scenario"]["product_id"]
store_id = demo["scenario"]["store_id"]
context = service.analysis(product_id, store_id)["context"]
display(pd.Series(context).to_frame("value"))""",
        ),
        (
            "md",
            "## Reproduce the baseline and proposal\nThe week, store and seasonal conditions are held fixed. Costs are absent unless explicitly supplied. The model version identifies the fit used for the saved result.",
        ),
        (
            "code",
            """assumptions = demo["scenario"]["assumptions"]
fields = ScenarioRequest.model_fields
request = ScenarioRequest(**{key: value for key, value in assumptions.items() if key in fields})
comparison = service.simulate(request, persist=False)
assert comparison["model_version"] == demo["scenario"]["model_version"]
assert np.isclose(comparison["proposed"]["revenue"], demo["scenario"]["proposed"]["revenue"])
display(pd.DataFrame([comparison["baseline"], comparison["proposed"]], index=["baseline", "proposed"]))
display(pd.Series(comparison["changes"]).to_frame("simulated change"))""",
        ),
        (
            "code",
            """px.line(pd.DataFrame(comparison["curve"]), x="price", y="revenue", markers=True, title="Revenue curve at proposed support settings").show(renderer="plotly_mimetype")
print(comparison["disclaimer"])""",
        ),
        (
            "md",
            "## Search the feasible grid\nThe grid is small enough to enumerate. A candidate may improve modeled revenue but fail a margin constraint. A boundary result is expected for many constant-elasticity models.",
        ),
        (
            "code",
            """recommendation = service.recommend(RecommendationRequest(product_id=product_id, store_id=store_id))
print("Status:", recommendation["status"])
display(pd.DataFrame(recommendation["candidates"]).head(10))
print(recommendation["note"])""",
        ),
        (
            "md",
            "## A deliberate infeasible case\nThe following is a clearly labeled hypothetical high cost, not a value from the dataset. The tool should report no feasible recommendation rather than relaxing the constraint.",
        ),
        (
            "code",
            """high_cost = context["price_support"][1] * 2
infeasible = service.recommend(RecommendationRequest(product_id=product_id, store_id=store_id, unit_cost=high_cost, min_margin_pct=20))
assert infeasible["status"] == "infeasible"
print(infeasible["status"], "with assumed unit cost", high_cost)""",
        ),
        (
            "md",
            "## Decision memo\nExplain the proposed price/support, the allowed price range, simulated revenue change, uncertainty and assumptions. Do not equate support associations with experimental lift. A controlled trial and inventory checks would be needed before operational use.",
        ),
    ],
)
print("Created three analysis notebooks.")
