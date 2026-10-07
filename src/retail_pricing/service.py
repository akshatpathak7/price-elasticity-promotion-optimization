"""Business calculations shared by the API, notebooks and tests."""

import json
from datetime import timedelta

import pandas as pd
from sqlalchemy import select

from . import db
from .config import ARTIFACTS
from .data import PROMOTIONS
from .modeling import load_artifacts, predict_elasticity

DISCLAIMER = (
    "Observational model estimates, not causal effects or realized gains. "
    "Intervals describe conditional mean uncertainty, not a range of future sales."
)


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def outcome(price, units, lower, upper, unit_cost=None, promotion_cost=0):
    revenue = price * units
    profit = (price - unit_cost) * units - promotion_cost if unit_cost is not None else None
    return {
        "price": float(price),
        "units": float(units),
        "revenue": float(revenue),
        "units_interval": [float(lower), float(upper)],
        "revenue_interval": [float(price * lower), float(price * upper)],
        "contribution_profit": float(profit) if profit is not None else None,
        "margin_pct": float(100 * profit / revenue) if profit is not None and revenue else None,
    }


class PricingService:
    def __init__(self, engine, artifacts=None, summary=None):
        self.engine = engine
        if artifacts is None:
            artifacts, summary = load_artifacts()
        self.models = artifacts["models"]
        self.summary = summary
        quality = db.check(engine)
        if quality["dataset_id"] != summary["dataset_id"]:
            raise ValueError("Dataset and model versions differ. Train models for this dataset.")
        self.quality = quality
        prediction_file = ARTIFACTS / summary["version"] / "predictions.parquet"
        self.predictions = pd.read_parquet(prediction_file) if prediction_file.exists() else None

    def catalog(self):
        with self.engine.connect() as conn:
            products = pd.read_sql(select(db.products), conn)
        products = products[products.product_id.isin(self.models)].copy()
        products["eligible"] = products.product_id.map(
            lambda p: not self.summary["diagnostics"][p]["ineligibility_reasons"]
        )
        return {
            "model_version": self.summary["version"],
            "products": records(products),
            "quality": self.quality,
            "next_week": self.next_week.date().isoformat(),
        }

    @property
    def next_week(self):
        return pd.Timestamp(
            pd.Timestamp(self.summary["latest_week"]).to_pydatetime() + timedelta(days=7)
        )

    def history(self, product_id):
        if product_id not in self.models:
            raise LookupError("Product has no fitted model")
        with self.engine.connect() as conn:
            frame = pd.read_sql(
                select(db.observations).where(db.observations.c.product_id == product_id),
                conn,
                parse_dates=["week"],
            )
        return frame.sort_values(["store_id", "week"])

    def context(self, product_id, store_id):
        history = self.history(product_id)
        local = history[history.store_id == store_id]
        if local.empty or store_id not in self.models[product_id]["stores"]:
            raise LookupError("Store has no trained history for this product")
        latest = local.iloc[-1]
        lower, upper = local.price.quantile([0.05, 0.95]).tolist()
        combos = sorted(set(history[PROMOTIONS].itertuples(index=False, name=None)))
        warnings = list(self.summary["diagnostics"][product_id]["ineligibility_reasons"])
        train = history[history.week <= pd.Timestamp(self.summary["training_end"])]
        trained_combos = set(train[PROMOTIONS].itertuples(index=False, name=None))
        if tuple(int(latest[c]) for c in PROMOTIONS) not in trained_combos:
            warnings.append("Baseline support combination absent from training.")
        if local.week.nunique() < 78 or local.price.round(2).nunique() < 5:
            warnings.append("Insufficient store-level history or price variation.")
        if (pd.Timestamp(self.summary["latest_week"]) - latest.week).days > 14:
            warnings.append("Latest store observation is more than two weeks old.")
        return history, local, latest, lower, upper, combos, warnings

    def analysis(self, product_id, store_id=None):
        history = self.history(product_id)
        with self.engine.connect() as conn:
            store_lookup = pd.read_sql(select(db.stores), conn)
        available = store_lookup[
            store_lookup.store_id.isin(history.store_id)
            & store_lookup.store_id.isin(self.models[product_id]["stores"])
        ]
        view = history if store_id is None else history[history.store_id == store_id]
        if view.empty:
            raise LookupError("No observations for the selected store")
        weekly = view.groupby("week", as_index=False).agg(
            units=("units", "sum"),
            revenue=("revenue", "sum"),
            price=("price", "mean"),
            base_price=("base_price", "mean"),
            feature=("feature", "mean"),
            display=("display", "mean"),
            tpr_only=("tpr_only", "mean"),
        )
        result = {
            "model_version": self.summary["version"],
            "product_id": product_id,
            "diagnostics": self.summary["diagnostics"][product_id],
            "stores": records(available),
            "history": records(weekly),
            "metrics": {
                s: self.summary["metrics"][s]["by_product"].get(product_id)
                for s in ["validation", "test"]
            },
            "totals": {
                "units": float(view.units.sum()),
                "revenue": float(view.revenue.sum()),
                "rows": len(view),
                "promotion_share": float(view[PROMOTIONS].max(axis=1).mean()),
            },
            "disclaimer": DISCLAIMER,
        }
        result["prediction_history"] = []
        if self.predictions is not None:
            predicted = self.predictions[self.predictions.product_id == product_id]
            if store_id is not None:
                predicted = predicted[predicted.store_id == store_id]
            result["prediction_history"] = records(
                predicted.groupby("week", as_index=False)[
                    ["units", "baseline", "log_log", "xgboost"]
                ].sum()
            )
        if store_id is not None:
            _, _, latest, lo, hi, combos, warnings = self.context(product_id, store_id)
            result["context"] = {
                "baseline_price": float(latest.price),
                "reference_price": float(latest.base_price),
                "baseline_support": {c: int(latest[c]) for c in PROMOTIONS},
                "last_observed_week": latest.week.date().isoformat(),
                "price_support": [float(lo), float(hi)],
                "support_combinations": [
                    dict(zip(PROMOTIONS, map(int, c), strict=True)) for c in combos
                ],
                "next_week": self.next_week.date().isoformat(),
                "warnings": warnings,
            }
        return result

    def simulate(self, request, persist=True):
        history, local, latest, lo, hi, combos, warnings = self.context(
            request.product_id, request.store_id
        )
        reference = (
            request.reference_price
            if request.reference_price is not None
            else float(latest.base_price)
        )
        price = (
            reference * (1 - request.discount_pct / 100)
            if request.discount_pct is not None
            else request.price
        )
        price = round(price if price is not None else float(latest.price), 2)
        if price <= 0:
            raise ValueError("Rounded shelf price must be positive")
        support = (
            request.support.model_dump()
            if request.support
            else {c: int(latest[c]) for c in PROMOTIONS}
        )
        combo = tuple(support[c] for c in PROMOTIONS)
        if combo not in combos:
            raise ValueError("Promotional support combination was not observed for this product")
        if not lo - 0.005 <= price <= hi + 0.005 and abs(price - latest.price) > 0.005:
            raise ValueError(f"Price outside observed store support ({lo:.2f} to {hi:.2f})")
        if support["tpr_only"] and price >= reference:
            raise ValueError("Tag-only support requires a price below the reference price")
        train = history[history.week <= pd.Timestamp(self.summary["training_end"])]
        if not ((train[PROMOTIONS] == pd.Series(support)).all(axis=1)).any():
            warnings.append("Support combination absent from training; exploratory estimate only.")
        base_support = {c: int(latest[c]) for c in PROMOTIONS}
        scenario_frame = pd.DataFrame(
            [
                {
                    "week": self.next_week,
                    "store_id": request.store_id,
                    "price": latest.price,
                    **base_support,
                },
                {"week": self.next_week, "store_id": request.store_id, "price": price, **support},
                {
                    "week": self.next_week,
                    "store_id": request.store_id,
                    "price": price,
                    **dict.fromkeys(PROMOTIONS, 0),
                },
            ]
        )
        mean, lower, upper = predict_elasticity(
            self.models[request.product_id], scenario_frame, True
        )
        baseline = outcome(
            float(latest.price),
            mean[0],
            lower[0],
            upper[0],
            request.unit_cost,
            request.baseline_promotion_cost,
        )
        proposed = outcome(
            price, mean[1], lower[1], upper[1], request.unit_cost, request.promotion_cost
        )
        grid = sorted(
            set(
                [round(float(latest.price) * i / 100, 2) for i in range(80, 111)]
                + [float(latest.price)]
            )
        )
        grid = [p for p in grid if lo - 0.005 <= p <= hi + 0.005 or abs(p - latest.price) < 0.005]
        grid = [p for p in grid if not support["tpr_only"] or p < reference]
        curve_frame = pd.DataFrame(
            [
                {"week": self.next_week, "store_id": request.store_id, "price": p, **support}
                for p in grid
            ]
        )
        curve = []
        if grid:
            q = predict_elasticity(self.models[request.product_id], curve_frame)
            curve = [
                {"price": p, "units": float(u), "revenue": float(p * u)}
                for p, u in zip(grid, q, strict=True)
            ]
        no_support_seen = (0, 0, 0) in set(train[PROMOTIONS].itertuples(index=False, name=None))
        result = {
            "model_version": self.summary["version"],
            "dataset_id": self.summary["dataset_id"],
            "product_id": request.product_id,
            "store_id": request.store_id,
            "week": self.next_week.date().isoformat(),
            "baseline": baseline,
            "proposed": proposed,
            "changes": {
                "units_pct": float(100 * (mean[1] / mean[0] - 1)),
                "revenue_pct": float(100 * (proposed["revenue"] / baseline["revenue"] - 1)),
                "revenue": float(proposed["revenue"] - baseline["revenue"]),
            },
            "support_lift_at_same_price_pct": float(100 * (mean[1] / mean[2] - 1))
            if no_support_seen
            else None,
            "assumptions": {
                **request.model_dump(),
                "resolved_price": price,
                "reference_price": reference,
                "resolved_discount_pct": 100 * (1 - price / reference),
                "resolved_support": support,
                "baseline_support": base_support,
                "last_observed_week": latest.week.date().isoformat(),
                "costs_are_user_assumptions": True,
            },
            "eligible": not warnings,
            "warnings": warnings,
            "curve": curve,
            "disclaimer": DISCLAIMER,
        }
        if persist:
            result["scenario_id"] = db.save_scenario(self.engine, request.model_dump(), result)
        return result

    def recommend(self, request):
        history, local, latest, lo, hi, combos, warnings = self.context(
            request.product_id, request.store_id
        )
        train = history[history.week <= pd.Timestamp(self.summary["training_end"])]
        trained_combos = set(train[PROMOTIONS].itertuples(index=False, name=None))
        prices = sorted(
            set(
                [round(float(latest.price) * i / 100, 2) for i in range(80, 111)]
                + [float(latest.price)]
            )
        )
        candidates = []
        baseline_support = tuple(int(latest[c]) for c in PROMOTIONS)
        for price in prices:
            if not (lo - 0.005 <= price <= hi + 0.005 or abs(price - latest.price) < 0.005):
                continue
            if 100 * (1 - price / latest.base_price) > request.max_discount_pct + 1e-8:
                continue
            for combo in combos:
                if combo not in trained_combos or (combo[2] and price >= latest.base_price):
                    continue
                candidates.append(
                    {
                        "week": self.next_week,
                        "store_id": request.store_id,
                        "price": price,
                        **dict(zip(PROMOTIONS, combo, strict=True)),
                    }
                )
        ranked = []
        if candidates:
            frame = pd.DataFrame(candidates)
            mean, lower, upper = predict_elasticity(self.models[request.product_id], frame, True)
            for c, q, low, high in zip(candidates, mean, lower, upper, strict=True):
                combo = tuple(c[p] for p in PROMOTIONS)
                unchanged = abs(c["price"] - latest.price) < 0.005 and combo == baseline_support
                cost = (
                    request.baseline_promotion_cost
                    if unchanged
                    else (request.promotion_cost if any(combo) else 0)
                )
                value = outcome(c["price"], q, low, high, request.unit_cost, cost)
                if (
                    request.min_margin_pct is not None
                    and value["margin_pct"] < request.min_margin_pct
                ):
                    continue
                value["support"] = {p: int(c[p]) for p in PROMOTIONS}
                value["unchanged_baseline"] = bool(unchanged)
                value["promotion_cost"] = cost
                ranked.append(value)
        ranked.sort(
            key=lambda v: (
                -v["revenue"],
                not v["unchanged_baseline"],
                v["price"],
                tuple(v["support"].values()),
            )
        )
        status = (
            "infeasible" if not ranked else ("insufficient_evidence" if warnings else "recommended")
        )
        baseline_frame = pd.DataFrame(
            [
                {
                    "week": self.next_week,
                    "store_id": request.store_id,
                    "price": latest.price,
                    **dict(zip(PROMOTIONS, baseline_support, strict=True)),
                }
            ]
        )
        q, low, high = predict_elasticity(self.models[request.product_id], baseline_frame, True)
        baseline = outcome(
            latest.price, q[0], low[0], high[0], request.unit_cost, request.baseline_promotion_cost
        )
        best = ranked[0] if status == "recommended" else None
        return {
            "model_version": self.summary["version"],
            "dataset_id": self.summary["dataset_id"],
            "status": status,
            "baseline": baseline,
            "recommendation": best,
            "simulated_revenue_change_pct": 100 * (best["revenue"] / baseline["revenue"] - 1)
            if best
            else None,
            "candidates": ranked,
            "candidate_count": len(ranked),
            "warnings": warnings,
            "assumptions": request.model_dump(),
            "week": self.next_week.date().isoformat(),
            "note": "Best within the tested price grid. Constant elasticity often selects a boundary; this is not a universal optimum.",
            "disclaimer": DISCLAIMER,
        }
