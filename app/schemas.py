from pydantic import BaseModel, conlist, confloat
from enum import Enum
from typing import List, Literal

class CompanySize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"

class PredictionRequest(BaseModel):
    financial_ratios: conlist(
        confloat(ge=0, le=100),
        min_items=64,
        max_items=64
    )
    company_size: CompanySize

class PredictionResponse(BaseModel):
    probability: float
    risk: Literal["low", "medium", "high"]
    confidence: float
    metrics: dict
    warnings: List[str]