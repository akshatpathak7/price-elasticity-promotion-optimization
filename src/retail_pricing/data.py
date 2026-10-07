"""Download, validate and normalize the publisher's workbook without altering it."""

import hashlib
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .config import DATA, REPORTS

SOURCE_URL = (
    "https://downloads.ctfassets.net/psj0p18eh7z1/7B4xI3FKcgMLak9Wm95IHC/"
    "3b63fe3d37bc7122d13524514be4ab51/dunnhumby_Breakfast-at-the-Frat.zip"
)
PROMOTIONS = ["feature", "display", "tpr_only"]
KEY = ["product_id", "store_id", "week"]
RENAME = {"upc": "product_id", "store_num": "store_id", "week_end_date": "week", "spend": "revenue"}


def download():
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "breakfast.zip"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        with httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temp.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (raw / member.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(raw)
    return next(raw.rglob("*.xlsx"))


def validate(frame):
    """Fail on ambiguous keys; quarantine malformed records with explicit reasons."""
    df = frame.copy()
    required = KEY + ["units", "price", "base_price", "revenue"] + PROMOTIONS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    if df.duplicated(KEY).any():
        raise ValueError("Duplicate product/store/week keys; source correction required")
    reasons = pd.Series("", index=df.index)
    checks = {"missing_key": df[KEY].isna().any(axis=1)}
    for col in ["price", "base_price", "units", "revenue"] + PROMOTIONS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        checks[f"invalid_{col}"] = ~np.isfinite(df[col])
        if col in ["price", "base_price"]:
            checks[f"invalid_{col}"] |= df[col] <= 0
        elif col in PROMOTIONS:
            checks[f"invalid_{col}"] |= ~df[col].isin([0, 1])
        else:
            checks[f"invalid_{col}"] |= df[col] < 0
    checks["fractional_units"] = df.units.notna() & (df.units % 1 != 0)
    checks["inconsistent_support"] = (df.tpr_only == 1) & ((df.feature == 1) | (df.display == 1))
    for name, mask in checks.items():
        reasons.loc[mask] += name + ";"
    rejected = df.loc[reasons != ""].copy()
    rejected["reason"] = reasons[reasons != ""]
    clean = df.loc[reasons == ""].copy()
    clean["product_id"] = clean.product_id.astype(str)
    clean["store_id"] = clean.store_id.astype(str)
    clean = clean.sort_values(KEY).reset_index(drop=True)
    weeks = pd.date_range(clean.week.min(), clean.week.max(), freq="7D")
    if not clean.week.isin(weeks).all():
        raise ValueError("Dates are not aligned to a weekly calendar")
    counts = clean.groupby(["product_id", "store_id"]).size()
    report = {
        "source_rows": len(df),
        "clean_rows": len(clean),
        "rejected_rows": len(rejected),
        "rejection_reasons": {k: int(v.sum()) for k, v in checks.items() if v.any()},
        "zero_sales_retained": int((clean.units == 0).sum()),
        "products": int(clean.product_id.nunique()),
        "stores": int(clean.store_id.nunique()),
        "weeks": len(weeks),
        "first_week": str(weeks.min().date()),
        "last_week": str(weeks.max().date()),
        "missing_pair_weeks": int((len(weeks) - counts).sum()),
        "missing_week_definition": "Absent weeks within observed product/store pairs across full calendar; not zero-filled.",
        "price_above_base_rows": int((clean.price > clean.base_price).sum()),
        "revenue_rounding_difference_rows": int(
            (abs(clean.revenue - clean.price * clean.units) > 0.05).sum()
        ),
    }
    return clean, rejected, report


def ingest(workbook=None):
    workbook = Path(workbook) if workbook else download()
    df = pd.read_excel(
        workbook, sheet_name="dh Transaction Data", header=1, usecols="A:L", engine="calamine"
    )
    df.columns = df.columns.str.lower()
    clean, rejected, report = validate(df.rename(columns=RENAME))
    products = pd.read_excel(
        workbook, sheet_name="dh Products Lookup", header=1, usecols="A:F", engine="calamine"
    )
    products.columns = products.columns.str.lower()
    products = products.rename(columns={"upc": "product_id"})
    products["product_id"] = products.product_id.astype(str)
    stores = pd.read_excel(
        workbook, sheet_name="dh Store Lookup", header=1, usecols="A:I", engine="calamine"
    )
    stores.columns = stores.columns.str.lower()
    stores["store_id"] = stores.store_id.astype(str)
    report["duplicate_store_lookup_rows"] = int(stores.store_id.duplicated().sum())
    # Two source stores have conflicting segments. Preserve ambiguity, not an arbitrary tier.
    report["conflicting_store_segments"] = (
        stores.groupby("store_id").seg_value_name.nunique().loc[lambda x: x > 1].index.tolist()
    )
    for store_id in report["conflicting_store_segments"]:
        stores.loc[stores.store_id == store_id, "seg_value_name"] = "CONFLICTING SOURCE VALUES"
    stores = stores.drop_duplicates()
    for lookup, key in [(products, "product_id"), (stores, "store_id")]:
        if lookup[key].duplicated().any() or not clean[key].isin(lookup[key]).all():
            raise ValueError(f"Invalid {key} lookup")
    report["dataset_id"] = hashlib.sha256(workbook.read_bytes()).hexdigest()[:16]
    report["source_url"] = SOURCE_URL
    for directory in [DATA / "processed", REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("observations", clean),
        ("products", products),
        ("stores", stores),
        ("rejected", rejected),
    ]:
        frame.to_parquet(DATA / "processed" / f"{name}.parquet", index=False)
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    return report
