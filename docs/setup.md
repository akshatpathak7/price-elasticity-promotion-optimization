# Setup and development guide

A local retail workbench for understanding price sensitivity and comparing next-week price and promotion decisions. It combines an explainable demand model, a predictive challenger, and a constrained price search with a working dashboard and API.

**Business question:** For one product in one store, which historically supported price and promotion should a category manager investigate next week?

This is an educational analysis of historical retail data. All recommended outcomes are simulations, not realized gains or proven causal effects.

## What you can do

- Explore weekly sales, prices, promotion frequency, and data quality.
- Interpret product elasticity with confidence intervals and store-specific demand levels.
- Compare a shelf price or discount with an unchanged-price baseline under the same calendar conditions.
- Enter optional cost assumptions and constrain discounts and contribution margins.
- Evaluate a bounded price grid and see why a recommendation is available, unsupported, or infeasible.
- Export comparisons and retrieve saved scenarios with their model version and assumptions.

## Quick start: Docker

Requires Docker with Compose and a running engine. Ports 5433, 8000, and 8501 must be available. The database credentials are local demonstration defaults, and published ports bind only to localhost.

```bash
docker compose build
docker compose up -d db
docker compose run --rm pipeline
docker compose up -d api dashboard
```

Open [the dashboard](http://localhost:8501) and [interactive API documentation](http://localhost:8000/docs). The pipeline downloads the source workbook, validates it, loads PostgreSQL, trains the models, and creates a measured demonstration. Initial setup needs internet access; subsequent dashboard use is local.

```bash
docker compose logs api dashboard
docker compose down
```

Stopping services preserves data. After explicitly retraining with the pipeline, restart the API so it loads the new model run.

## Local Python development

Commands install the package without editable path files for portability. Changes under `src` invalidate the cached build automatically. Run commands from the project root.

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). PostgreSQL must be running; either use the Compose database or the native commands below.

```bash
uv sync --frozen --no-editable
cp .env.example .env
docker compose up -d db
uv run --no-editable retail-pricing setup
```

Start these in separate terminals from the project root:

```bash
uv run --no-editable uvicorn retail_pricing.api:app --host 127.0.0.1 --port 8000
uv run --no-editable streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

### Native PostgreSQL alternative

If PostgreSQL tools are installed but Docker is unavailable, initialize a dedicated local cluster once. Do not initialize over an existing cluster. This example trusts connections on localhost for a private local demo.

```bash
mkdir -p .runtime
initdb -D .runtime/postgres -U retail --auth=trust --encoding=UTF8
pg_ctl -D .runtime/postgres -l .runtime/postgres.log -o '-p 5433 -h 127.0.0.1' start
createdb -h 127.0.0.1 -p 5433 -U retail retail
uv run --no-editable retail-pricing setup
```

Later starts need only the `pg_ctl ... start` command, followed by the API and dashboard commands. Stop the cluster with `pg_ctl -D .runtime/postgres stop`. Do not run the native cluster and Compose database on port 5433 simultaneously.

On macOS, if XGBoost reports a missing OpenMP library, install `libomp` with Homebrew. The Linux container includes its OpenMP runtime.

## Pipeline commands

| Command | Responsibility |
|---|---|
| `uv run --no-editable retail-pricing fetch` | Download the publisher archive and preserve the workbook |
| `uv run --no-editable retail-pricing ingest` | Normalize data; write cleaned Parquet and a quality report |
| `uv run --no-editable retail-pricing load` | Transactionally replace observations and lookups in PostgreSQL |
| `uv run --no-editable retail-pricing train` | Fit, evaluate, register, and activate a new immutable model run |
| `uv run --no-editable retail-pricing demo` | Save an eligible real-data scenario and demonstration report |
| `uv run --no-editable retail-pricing setup` | Run ingestion, database loading, training, and demonstration |

`ingest --workbook /path/to/workbook.xlsx` uses an existing publisher workbook. Stop the API before replacing data or retraining, and restart afterward. Never load untrusted model artifact files.

## Evidence and design

The source workbook contains 524,950 rows over 156 weeks. The current importer retains 524,741 rows for 55 observed products and 77 distinct stores. The product lookup has 58 products; the store lookup has 79 rows but two duplicate IDs with conflicting segment labels. See [data documentation](data.md).

Models fit the first 104 weeks. Validation uses the next 26; the final 26 are held out. The statistical model is intentionally frozen after training so its reported performance and scenario coefficients refer to the same fit. The demonstration week is **2012-01-11**, not the current week.

On the measured full-data run, test WAPE was approximately 40.48% for the log–log model, 43.37% for the rolling baseline, and 47.26% for XGBoost. More complexity did not improve this test result. See [the measured results](results.md) for exact metrics and interpretation.

The rolling baseline updates from prior observed sales each week. Models are evaluated on common rows for a fair comparison, with excluded rows reported. Accuracy at historical prices is not evidence that a proposed intervention will achieve its estimated lift.

## Project map

```text
src/retail_pricing/   ingestion, persistence, modeling, scenarios, API, CLI
dashboard/          Streamlit client
notebooks/          exploration, model comparison, decision walkthrough
tests/              data, leakage, model, business-rule, API and dashboard tests
docs/               methodology, architecture, data, results and demo guide
data/raw/           unchanged publisher files (not committed)
data/processed/     validated Parquet (not committed)
artifacts/          versioned fitted models and prediction records (not committed)
reports/            quality, metrics and measured demo outputs
```

## Verification

```bash
uv run --no-editable pytest -q
uv run --no-editable ruff check src tests dashboard
```

Tests use a clearly synthetic fixture with known elasticity and isolated SQLite storage for fast unit/API tests. Production persistence uses PostgreSQL. The optional end-to-end check uses the running local API:

```bash
uv run --no-editable python scripts/verify_local.py
```

Register the project kernel once:

```bash
uv run --no-editable python -m ipykernel install --prefix .venv --name retail-pricing --display-name 'Retail Pricing'
```

Open the three notebooks with this kernel; run from top to bottom after setup. They reuse project functions rather than duplicate the pipeline.

## Reading order

1. [Five-minute demonstration](demo.md)
2. [Methodology and limitations](methodology.md)
3. [Architecture and API contracts](architecture.md)
4. [Data dictionary and provenance](data.md)
5. [Measured results](results.md)

The first version excludes cross-product substitution, inventory constraints, carryover effects, causal identification, authentication, and public deployment. Those require additional data or scope, rather than additional labels on the same estimates.
