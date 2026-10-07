"""Explicit pipeline commands: fetch, ingest, load, train, and demo."""

import argparse
import json
from pathlib import Path

from . import db
from .config import REPORTS
from .data import download, ingest
from .modeling import activate, train_all
from .schemas import RecommendationRequest, ScenarioRequest
from .service import PricingService


def train():
    summary = train_all()
    database = db.engine()
    db.initialize(database)
    db.save_run(database, summary)
    activate(summary["version"])
    print(
        json.dumps(
            {"version": summary["version"], "test_metrics": summary["metrics"]["test"]["overall"]},
            indent=2,
        )
    )


def demo():
    service = PricingService(db.engine())
    eligible = [
        p for p, d in service.summary["diagnostics"].items() if not d["ineligibility_reasons"]
    ]
    for product in eligible:
        history = service.history(product)
        stores = history.groupby("store_id").size().sort_values(ascending=False).index
        for store in stores:
            result = service.recommend(RecommendationRequest(product_id=product, store_id=store))
            if result["status"] == "recommended":
                best = result["recommendation"]
                comparison = service.simulate(
                    ScenarioRequest(
                        product_id=product,
                        store_id=store,
                        price=best["price"],
                        support=best["support"],
                    )
                )
                payload = {"recommendation": result, "scenario": comparison}
                (REPORTS / "demo_result.json").write_text(json.dumps(payload, indent=2))
                print(
                    json.dumps(
                        {
                            "product_id": product,
                            "store_id": store,
                            "scenario_id": comparison["scenario_id"],
                            "changes": comparison["changes"],
                        },
                        indent=2,
                    )
                )
                return
    raise RuntimeError(
        "No eligible product/store found. Inspect diagnostics; do not relax gates silently."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "ingest", "load", "train", "setup", "demo"])
    parser.add_argument("--workbook", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        print(download())
    elif args.command == "ingest":
        print(json.dumps(ingest(args.workbook), indent=2))
    elif args.command == "load":
        print(json.dumps(db.load_data(db.engine()), indent=2))
    elif args.command == "train":
        train()
    elif args.command == "demo":
        demo()
    else:
        print(json.dumps(ingest(args.workbook), indent=2))
        db.load_data(db.engine())
        train()
        demo()


if __name__ == "__main__":
    main()
