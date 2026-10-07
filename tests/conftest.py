import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from retail_pricing import db
from retail_pricing.modeling import fit_elasticity
from retail_pricing.service import PricingService


@pytest.fixture(scope="session")
def sales():
    rng = np.random.default_rng(42)
    rows = []
    for store in ["1", "2", "3", "4"]:
        for t, week in enumerate(pd.date_range("2009-01-14", periods=156, freq="7D")):
            price = float(rng.choice(np.arange(1.5, 3.01, 0.1)))
            display = int(rng.integers(0, 2))
            units = np.exp(
                4
                + 0.1 * int(store)
                - 1.6 * np.log(price)
                + 0.25 * display
                + 0.1 * np.sin(2 * np.pi * t / 52.18)
                + rng.normal(0, 0.025)
            )
            rows.append(
                dict(
                    product_id="p1",
                    store_id=store,
                    week=week,
                    price=price,
                    base_price=3.0,
                    units=round(units),
                    revenue=round(units) * price,
                    feature=0,
                    display=display,
                    tpr_only=0,
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def fitted(sales):
    train = sales[sales.week <= sorted(sales.week.unique())[103]]
    return fit_elasticity(train, sales.week.min())


@pytest.fixture
def service(sales, fitted):
    import copy

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.initialize(engine)
    with engine.begin() as conn:
        sales.to_sql("observations", conn, if_exists="append", index=False)
        conn.execute(
            db.settings.insert().values(
                key="quality",
                value={
                    "dataset_id": "synthetic",
                    "first_week": "2009-01-14",
                    "last_week": "2012-01-04",
                    "clean_rows": len(sales),
                    "rejected_rows": 0,
                    "missing_pair_weeks": 0,
                },
            )
        )
        conn.execute(
            db.products.insert().values(
                product_id="p1", description="Test product", category="Test"
            )
        )
        conn.execute(
            db.stores.insert(), [{"store_id": s, "store_name": s} for s in ["1", "2", "3", "4"]]
        )
    model = copy.deepcopy(fitted)
    model["diagnostics"]["ineligibility_reasons"] = []
    summary = {
        "version": "test-model",
        "dataset_id": "synthetic",
        "latest_week": str(sales.week.max().date()),
        "training_end": str(sorted(sales.week.unique())[103].date()),
        "diagnostics": {"p1": model["diagnostics"]},
        "metrics": {
            "validation": {"by_product": {}, "overall": {}},
            "test": {"by_product": {}, "overall": {}},
        },
    }
    for split in summary["metrics"].values():
        scores = {
            name: {"mae": 1.0, "wape": 0.1, "bias": 0.0, "rows": 104}
            for name in ["baseline", "log_log", "xgboost"]
        }
        split["overall"] = scores
        split["by_product"]["p1"] = scores
    yield PricingService(engine, {"models": {"p1": model}}, summary)
    engine.dispose()
