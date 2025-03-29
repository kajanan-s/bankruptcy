from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from enum import Enum
from typing import List, Literal
import numpy as np
import joblib
from .config import settings

app = FastAPI()
security = HTTPBearer()

# Load model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class CompanySize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"

class PredictionRequest(BaseModel):
    financial_ratios: List[float]
    company_size: CompanySize

class PredictionResponse(BaseModel):
    probability: float
    risk: Literal["low", "medium", "high"]
    confidence: float
    metrics: dict
    warnings: List[str]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.financial_ratios) != 64:
            raise HTTPException(
                status_code=422,
                detail="Exactly 64 financial ratios required"
            )
        
        # Prepare features
        features = np.array([request.financial_ratios])
        
        # Predict
        proba = model.predict_proba(features)[0][1]
        
        return {
            "probability": float(proba),
            "risk": "high" if proba > 0.7 else "medium" if proba > 0.3 else "low",
            "confidence": min(proba * 1.5, 0.99),
            "metrics": model.metrics if hasattr(model, 'metrics') else {},
            "warnings": [
                "Trained on historical data (2007-2019)",
                "Not validated for financial institutions"
            ]
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))