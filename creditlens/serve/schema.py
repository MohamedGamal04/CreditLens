"""Pydantic request/response models for the /predict API.

The request mirrors the frontend applicant form (app/src/model.js DEFAULT_APPLICANT);
``contract`` / ``education`` are accepted but unused by the 15-feature model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Model keys the API accepts. 'stack' is the frontend's alias for 'stacking'.
ModelKey = Literal["logreg", "rf", "xgb", "lgbm", "catboost", "stack", "stacking"]


class Applicant(BaseModel):
    """Raw applicant fields collected by the form."""

    # application
    amt_income: float = Field(gt=0, examples=[162000])
    amt_credit: float = Field(gt=0, examples=[640000])
    amt_annuity: float = Field(gt=0, examples=[31500])
    age: float = Field(ge=18, le=100, examples=[36])
    emp_years: float = Field(ge=0, le=60, examples=[3.5])
    region_rating: int = Field(ge=1, le=3, examples=[2])
    cnt_children: int = Field(ge=0, examples=[1])
    ext_source_1: float = Field(ge=0, le=1, examples=[0.42])
    ext_source_2: float = Field(ge=0, le=1, examples=[0.48])
    ext_source_3: float = Field(ge=0, le=1, examples=[0.43])
    # bureau
    bureau_active: int = Field(ge=0, examples=[2])
    bureau_dpd: int = Field(ge=0, examples=[0])
    bureau_debt: float = Field(ge=0, examples=[165000])
    # previous_application
    prev_approval: float = Field(ge=0, le=1, examples=[0.58])
    prev_refused: int = Field(ge=0, examples=[1])
    prev_count: int = Field(ge=0, examples=[4])
    # optional / unused-by-model
    contract: str | None = None
    education: str | None = None


class PredictRequest(BaseModel):
    applicant: Applicant
    model: ModelKey = "lgbm"
    low: float | None = Field(None, ge=0, le=1, description="low/medium band cut")
    high: float | None = Field(None, ge=0, le=1, description="medium/high band cut")


class PredictResponse(BaseModel):
    probability: float = Field(description="calibrated probability of default (PD)")
    band: Literal["low", "med", "high"]
    decision: Literal["approve", "review", "decline"]
    model: str
    features: dict


class ExplainRequest(BaseModel):
    applicant: Applicant
    model: ModelKey = "lgbm"
    top_n: int = Field(8, ge=1, le=15)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float


class ExplainResponse(BaseModel):
    model: str
    probability: float = Field(description="calibrated PD (headline number)")
    available: bool = Field(description="False for the stacking ensemble")
    contributions: list[FeatureContribution]  # empty when unavailable
