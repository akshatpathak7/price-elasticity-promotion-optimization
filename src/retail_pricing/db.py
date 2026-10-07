"""Small relational persistence layer. Model coefficients stay in local artifacts."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    select,
    text,
)

from .config import DATA, REPORTS, database_url

metadata = MetaData()
observations = Table(
    "observations",
    metadata,
    Column("product_id", String, primary_key=True),
    Column("store_id", String, primary_key=True),
    Column("week", DateTime, primary_key=True),
    *[Column(c, Float, nullable=False) for c in ["units", "price", "base_price", "revenue"]],
    *[Column(c, Integer, nullable=False) for c in ["feature", "display", "tpr_only"]],
)
products = Table(
    "products",
    metadata,
    Column("product_id", String, primary_key=True),
    *[Column(c, String) for c in ["description", "manufacturer", "category", "product_size"]],
)
stores = Table(
    "stores",
    metadata,
    Column("store_id", String, primary_key=True),
    *[
        Column(c, String)
        for c in ["store_name", "address_city_name", "address_state_prov_code", "seg_value_name"]
    ],
)
settings = Table(
    "dataset", metadata, Column("key", String, primary_key=True), Column("value", JSON)
)
runs = Table(
    "model_runs",
    metadata,
    Column("version", String, primary_key=True),
    Column("created_at", DateTime(timezone=True)),
    Column("summary", JSON, nullable=False),
)
scenarios = Table(
    "scenarios",
    metadata,
    Column("id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True)),
    Column("model_version", String),
    Column("request", JSON, nullable=False),
    Column("result", JSON, nullable=False),
)


def engine():
    return create_engine(database_url(), pool_pre_ping=True)


def initialize(db):
    metadata.create_all(db)


def load_data(db):
    initialize(db)
    quality = json.loads((REPORTS / "data_quality.json").read_text())
    # One transaction: API readers never see a partially loaded dataset.
    with db.begin() as conn:
        for table in [observations, products, stores]:
            df = pd.read_parquet(DATA / "processed" / f"{table.name}.parquet")
            cols = list(table.columns.keys())
            conn.execute(delete(table))
            if db.dialect.name == "postgresql":
                with conn.connection.driver_connection.cursor() as cur:
                    with cur.copy(f"COPY {table.name} ({', '.join(cols)}) FROM STDIN") as copy:
                        for row in df[cols].itertuples(index=False, name=None):
                            copy.write_row(row)
            else:  # Isolated test databases only.
                df[cols].to_sql(table.name, conn, if_exists="append", index=False)
        conn.execute(delete(settings))
        conn.execute(settings.insert().values(key="quality", value=quality))
    return quality


def save_run(db, summary):
    with db.begin() as conn:
        conn.execute(
            runs.insert().values(
                version=summary["version"], created_at=datetime.now(timezone.utc), summary=summary
            )
        )


def save_scenario(db, request, result):
    scenario_id = str(uuid4())
    with db.begin() as conn:
        conn.execute(
            scenarios.insert().values(
                id=scenario_id,
                created_at=datetime.now(timezone.utc),
                model_version=result["model_version"],
                request=request,
                result=result,
            )
        )
    return scenario_id


def check(db):
    with db.connect() as conn:
        conn.execute(text("SELECT 1"))
        return conn.execute(
            select(settings.c.value).where(settings.c.key == "quality")
        ).scalar_one()
