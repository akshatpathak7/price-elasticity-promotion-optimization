"""Local HTTP interface; no fitting takes place in request handlers."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from . import db
from .schemas import RecommendationRequest, ScenarioRequest
from .service import PricingService

app = FastAPI(
    title="Retail Pricing Workbench",
    version="0.1.0",
    description="Historical demand analysis and one-week price scenarios.",
)


@lru_cache
def get_engine():
    return db.engine()


@lru_cache
def get_service():
    try:
        return PricingService(get_engine())
    except (OSError, ValueError, SQLAlchemyError) as exc:
        raise HTTPException(
            503, "Data or model unavailable. Run the setup pipeline and restart the API."
        ) from exc


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LookupError)
async def lookup_error(request, exc):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(SQLAlchemyError)
async def database_error(request, exc):
    return JSONResponse(
        status_code=503, content={"detail": "Database unavailable. Check the local service."}
    )


@app.get("/health")
def health(service: PricingService = Depends(get_service)):
    quality = db.check(service.engine)
    if quality["dataset_id"] != service.summary["dataset_id"]:
        raise HTTPException(503, "Dataset changed; retrain and restart the API.")
    return {"status": "ok", "model_version": service.summary["version"], "database": "connected"}


@app.get("/products")
def products(service: PricingService = Depends(get_service)):
    return service.catalog()


@app.get("/products/{product_id}/analysis")
def analysis(
    product_id: str, store_id: str | None = None, service: PricingService = Depends(get_service)
):
    return service.analysis(product_id, store_id)


@app.get("/models/metrics")
def model_metrics(service: PricingService = Depends(get_service)):
    return service.summary


@app.post("/scenarios")
def scenarios(request: ScenarioRequest, service: PricingService = Depends(get_service)):
    return service.simulate(request)


@app.get("/scenarios/{scenario_id}")
def saved_scenario(scenario_id: str, service: PricingService = Depends(get_service)):
    with service.engine.connect() as conn:
        row = (
            conn.execute(select(db.scenarios).where(db.scenarios.c.id == scenario_id))
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(404, "Scenario not found")
    return {**row["result"], "scenario_id": scenario_id}


@app.post("/recommendations")
def recommendations(request: RecommendationRequest, service: PricingService = Depends(get_service)):
    return service.recommend(request)
