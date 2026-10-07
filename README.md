# Price Elasticity & Promotion Optimization

A retail analytics project that estimates how demand responds to price changes and promotions. It helps compare next-week pricing scenarios and identify revenue-maximizing candidates for a product in a store, subject to discount and optional margin constraints.

## What it demonstrates

- **Data preparation:** cleaning and validating historical retail sales, prices, and promotion records.
- **Statistical modeling:** estimating price elasticity with interpretable log–log regression and confidence intervals.
- **Machine learning:** comparing the regression model with a rolling sales baseline and an XGBoost challenger.
- **Business decision-making:** simulating demand, revenue, and optional contribution profit under alternative prices and promotions.
- **Application development:** connecting a Streamlit dashboard, FastAPI service, and PostgreSQL database with reproducible pipelines and tests.

## How it works

1. Download the [dunnhumby Breakfast at the Frat dataset](https://www.dunnhumby.com/source-files/), validate the observations, and load cleaned data into PostgreSQL.
2. Train on the first 104 weeks, validate on the next 26, and evaluate on the final 26.
3. Fit product-level elasticity models with store effects, seasonality, trend, and promotional support indicators. Compare their forecasts with the baseline and XGBoost.
4. Use the dashboard to compare a proposed price or discount with the latest observed price under identical next-week conditions.
5. Search a bounded price grid for the best feasible revenue estimate. Return an explanation instead of a recommendation when evidence is insufficient.

The dashboard includes sales exploration, elasticity and promotion analysis, a scenario simulator, and model validation. Saved scenarios retain their assumptions and model version.

## Results

The cleaned dataset contains **524,741 observations across 55 products, 77 stores, and 156 weeks**. On the held-out test period:

| Model | MAE (units) | WAPE |
|---|---:|---:|
| Rolling sales baseline | 7.92 | 43.37% |
| Log–log regression | 7.40 | 40.48% |
| XGBoost | 8.63 | 47.26% |

The interpretable model reduced test MAE by approximately **6.7%** versus the baseline. See [measured results](docs/results.md) for details.

## Tech stack

Python, Pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, PostgreSQL, FastAPI, Streamlit, and Plotly.

## Run locally

With Docker and Docker Compose running:

```bash
git clone https://github.com/akshatpathak7/price-elasticity-promotion-optimization.git
cd price-elasticity-promotion-optimization
docker compose build
docker compose up -d db
docker compose run --rm pipeline
docker compose up -d api dashboard
```

Open the dashboard at **http://localhost:8501** and API documentation at **http://localhost:8000/docs**. Initial setup downloads the source data and trains the models. Raw data, local credentials, and fitted artifacts are not committed.

For a native Python/PostgreSQL setup, testing commands, and notebook instructions, see the [setup guide](docs/setup.md). The native workflow has been verified; Compose configuration has been validated, but container execution has not yet been verified.

## Explore the project

- [Analysis notebooks](notebooks): data exploration, model comparison, and pricing decisions.
- [Methodology](docs/methodology.md): modeling choices, evaluation, and optimization rules.
- [Architecture](docs/architecture.md): pipeline, storage, and API contracts.
- [Data documentation](docs/data.md): provenance, dictionary, and cleaning policy.
- [Demo walkthrough](docs/demo.md): a guided tour of the application.
- [Tests](tests): data checks, leakage prevention, model behavior, business rules, API persistence, and dashboard interactions.

## Limitations

The dataset represents historical retail activity; the demonstration simulates the week ending January 11, 2012. Pricing and promotion effects are observational estimates, not proof of causation. Simulated revenue changes are not realized gains. Product costs must be supplied as assumptions, and this version does not model inventory, competitor responses, or cross-product substitution.
