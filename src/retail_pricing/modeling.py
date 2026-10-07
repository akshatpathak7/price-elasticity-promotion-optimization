"""Explainable demand models, chronological evaluation and immutable model runs."""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .config import ARTIFACTS, DATA, REPORTS
from .data import PROMOTIONS

NUMERIC = ["log_price", "trend", "sin_year", "cos_year"] + PROMOTIONS


def features(df, origin):
    x = df.copy()
    elapsed = (pd.to_datetime(x.week) - pd.Timestamp(origin)).dt.days / 7
    x["log_price"] = np.log(x.price)
    x["trend"] = elapsed / 52
    x["sin_year"] = np.sin(2 * np.pi * elapsed / 52.18)
    x["cos_year"] = np.cos(2 * np.pi * elapsed / 52.18)
    return x


def split_calendar(df):
    weeks = pd.date_range(df.week.min(), df.week.max(), freq="7D")
    if len(weeks) != 156:
        raise ValueError(f"Expected a 156-week calendar; found {len(weeks)}")
    return {
        "train": df[df.week <= weeks[103]].copy(),
        "validation": df[(df.week >= weeks[104]) & (df.week <= weeks[129])].copy(),
        "test": df[df.week >= weeks[130]].copy(),
    }


def design(df, origin, columns=None):
    x = features(df, origin)
    numeric = x[NUMERIC].astype(float)
    store = pd.get_dummies(x.store_id.astype(str), prefix="store", dtype=float)
    # One store is the reference category, avoiding the dummy-variable trap.
    store = store.iloc[:, 1:]
    out = pd.concat([pd.Series(1.0, index=x.index, name="intercept"), numeric, store], axis=1)
    if columns is not None:
        # Build every store indicator explicitly before aligning: a single-row prediction
        # must not lose its store just because it is the first store in that request.
        for name in columns:
            if name.startswith("store_"):
                out[name] = (x.store_id.astype(str) == name[6:]).astype(float)
        return out.reindex(columns=columns, fill_value=0.0)
    return out


def fit_elasticity(train, origin):
    positive = train.loc[train.units > 0].copy()
    if len(positive) < 100 or positive.store_id.nunique() < 2:
        raise ValueError("Too few positive observations or stores to estimate elasticity")
    x = design(positive, origin)
    # A constant support flag has no independently estimable effect.
    active = [c for c in x if c == "intercept" or x[c].nunique() > 1]
    x = x[active]
    if "log_price" not in x:
        raise ValueError("No price variation")
    model = sm.OLS(np.log(positive.units), x).fit(
        cov_type="cluster", cov_kwds={"groups": positive.store_id}
    )
    ci = model.conf_int().loc["log_price"].tolist()
    rank = int(np.linalg.matrix_rank(x.to_numpy()))
    return {
        "columns": list(x.columns),
        "coefficients": model.params.to_numpy(),
        "covariance": model.cov_params().to_numpy(),
        "origin": str(pd.Timestamp(origin).date()),
        "smearing": float(np.exp(model.resid).mean()),
        "stores": sorted(positive.store_id.unique().tolist()),
        "diagnostics": {
            "elasticity": float(model.params["log_price"]),
            "elasticity_ci": [float(v) for v in ci],
            "r_squared_log": float(model.rsquared),
            "training_rows": len(positive),
            "zero_sales_excluded": int((train.units == 0).sum()),
            "rank_deficient": rank < x.shape[1],
            "condition_number": float(model.condition_number),
            "distinct_prices": int(positive.price.round(2).nunique()),
            "price_log_std": float(np.log(positive.price).std()),
            "training_weeks": int(positive.week.nunique()),
            "support_coefficients": {
                p: float(model.params[p]) for p in PROMOTIONS if p in model.params
            },
            "uncertainty_method": "95% interval, standard errors clustered by store",
        },
    }


def predict_elasticity(model, df, interval=False):
    if not df.store_id.astype(str).isin(model["stores"]).all():
        raise ValueError("Store was not represented in positive-sales training data")
    x = design(df, model["origin"], model["columns"]).to_numpy()
    log_mean = x @ model["coefficients"]
    mean = np.exp(log_mean) * model["smearing"]
    if not interval:
        return mean
    variance = np.einsum("ij,jk,ik->i", x, model["covariance"], x)
    half_width = norm.ppf(0.975) * np.sqrt(np.maximum(variance, 0))
    return (
        mean,
        np.exp(log_mean - half_width) * model["smearing"],
        np.exp(log_mean + half_width) * model["smearing"],
    )


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "wape": float(np.abs(actual - predicted).sum() / denominator) if denominator else None,
        "bias": float((predicted - actual).mean()),
        "rows": len(actual),
    }


def rolling_baseline(df):
    ordered = df.sort_values(["product_id", "store_id", "week"]).copy()
    ordered["baseline"] = ordered.groupby(["product_id", "store_id"]).units.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return ordered.baseline.reindex(df.index)


def eligibility(diagnostics, validation):
    reasons = []
    if (
        diagnostics["training_weeks"] < 78
        or diagnostics["distinct_prices"] < 5
        or diagnostics["price_log_std"] < 0.03
    ):
        reasons.append("Insufficient historical price variation or training coverage.")
    if diagnostics["rank_deficient"]:
        reasons.append("Model terms are not independently identifiable.")
    if not np.isfinite(diagnostics["elasticity_ci"]).all() or diagnostics["elasticity_ci"][1] >= 0:
        reasons.append("Elasticity is not confidently negative.")
    if not validation or validation["log_log"]["mae"] >= validation["baseline"]["mae"]:
        reasons.append("Statistical model did not beat the rolling baseline on validation MAE.")
    return reasons


def train_all():
    df = pd.read_parquet(DATA / "processed" / "observations.parquet")
    quality = json.loads((REPORTS / "data_quality.json").read_text())
    parts = split_calendar(df)
    origin = df.week.min()
    models, diagnostics, skipped = {}, {}, {}
    with threadpool_limits(limits=2):
        for product, group in parts["train"].groupby("product_id"):
            try:
                model = fit_elasticity(group, origin)
                models[product] = model
                diagnostics[product] = model["diagnostics"]
            except ValueError as exc:
                skipped[product] = str(exc)
        challenger = Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("numeric", "passthrough", NUMERIC),
                            (
                                "ids",
                                OneHotEncoder(handle_unknown="ignore"),
                                ["product_id", "store_id"],
                            ),
                        ]
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=180,
                        max_depth=6,
                        learning_rate=0.07,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        )
        challenger.fit(features(parts["train"], origin), parts["train"].units)
        df["baseline"] = rolling_baseline(df)
        results, predictions = {}, []
        for name in ["validation", "test"]:
            holdout = df.loc[parts[name].index].copy()
            holdout["log_log"] = np.nan
            for product, model in models.items():
                mask = (holdout.product_id == product) & holdout.store_id.isin(model["stores"])
                if mask.any():
                    holdout.loc[mask, "log_log"] = predict_elasticity(model, holdout.loc[mask])
            holdout["xgboost"] = np.maximum(0, challenger.predict(features(holdout, origin)))
            comparable = holdout.dropna(subset=["baseline", "log_log", "xgboost"]).copy()
            results[name] = {
                "overall": {
                    m: metrics(comparable.units, comparable[m])
                    for m in ["baseline", "log_log", "xgboost"]
                },
                "by_product": {
                    p: {m: metrics(g.units, g[m]) for m in ["baseline", "log_log", "xgboost"]}
                    for p, g in comparable.groupby("product_id")
                },
                "excluded_rows": len(holdout) - len(comparable),
                "first_week": str(holdout.week.min().date()),
                "last_week": str(holdout.week.max().date()),
            }
            comparable["split"] = name
            predictions.append(
                comparable[
                    [
                        "product_id",
                        "store_id",
                        "week",
                        "units",
                        "baseline",
                        "log_log",
                        "xgboost",
                        "split",
                    ]
                ]
            )
    for product in diagnostics:
        diagnostics[product]["ineligibility_reasons"] = eligibility(
            diagnostics[product], results["validation"]["by_product"].get(product)
        )
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ARTIFACTS / version
    directory.mkdir(parents=True)
    summary = {
        "version": version,
        "dataset_id": quality["dataset_id"],
        "metrics": results,
        "diagnostics": diagnostics,
        "skipped_products": skipped,
        "training_end": str(parts["train"].week.max().date()),
        "origin": str(origin.date()),
        "latest_week": str(df.week.max().date()),
        "model_policy": "Frozen first-104-week fit; validation gates recommendations; test is reporting only.",
    }
    joblib.dump({"models": models, "challenger": challenger}, directory / "models.joblib")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.concat(predictions).to_parquet(directory / "predictions.parquet", index=False)
    (REPORTS / "model_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def activate(version):
    ARTIFACTS.mkdir(exist_ok=True)
    temp = ARTIFACTS / "latest.tmp"
    temp.write_text(version)
    temp.replace(ARTIFACTS / "latest.txt")


def load_artifacts(version=None):
    version = version or (ARTIFACTS / "latest.txt").read_text().strip()
    directory = ARTIFACTS / version
    return joblib.load(directory / "models.joblib"), json.loads(
        (directory / "summary.json").read_text()
    )
