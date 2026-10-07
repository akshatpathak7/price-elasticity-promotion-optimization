"""Validated, documented request contracts for the decision engine."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: Literal[0, 1] = 0
    display: Literal[0, 1] = 0
    tpr_only: Literal[0, 1] = 0

    @model_validator(mode="after")
    def exclusive_tag(self):
        if self.tpr_only and (self.feature or self.display):
            raise ValueError("A tag-only promotion cannot also be featured or displayed")
        return self


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    product_id: str
    store_id: str
    price: float | None = Field(default=None, gt=0)
    reference_price: float | None = Field(default=None, gt=0)
    discount_pct: float | None = Field(default=None, ge=0, lt=100)
    support: Support | None = None
    unit_cost: float | None = Field(default=None, ge=0)
    promotion_cost: float = Field(default=0, ge=0, description="Assumed total weekly scenario cost")
    baseline_promotion_cost: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent_price(self):
        if self.price is not None and self.discount_pct is not None:
            raise ValueError("Supply either price or discount_pct, not both")
        return self


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    product_id: str
    store_id: str
    max_discount_pct: float = Field(default=20, ge=0, lt=100)
    unit_cost: float | None = Field(default=None, ge=0)
    promotion_cost: float = Field(default=0, ge=0)
    baseline_promotion_cost: float = Field(default=0, ge=0)
    min_margin_pct: float | None = Field(default=None, ge=0, lt=100)

    @model_validator(mode="after")
    def margin_requires_cost(self):
        if self.min_margin_pct is not None and self.unit_cost is None:
            raise ValueError("unit_cost is required for a minimum margin constraint")
        return self
