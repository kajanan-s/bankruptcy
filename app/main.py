from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Literal, Annotated
import numpy as np
import pandas as pd
import joblib
import time
import logging
from .config import settings
from .model import BankruptcyModel
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBearer()

try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

class CompanySize(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"

class PredictionRequest(BaseModel):
    financial_ratios: Annotated[List[float], Field(min_length=95, max_length=95, item_constraints={"ge": 0, "le": 100})]
    company_size: CompanySize

class PredictionResponse(BaseModel):
    probability: float
    risk: Literal["low", "medium", "high"]
    confidence: float
    metrics: dict
    warnings: List[str]
    latency_ms: float
    status: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    try:
        # Simulate time-series from 95 ratios
        ratio_columns = [f"ratio_{i+1}" for i in range(95)]
        long_df = pd.DataFrame({
            'id': [0] * 95,
            'time': ratio_columns,
            'value': request.financial_ratios
        })
        
        # Extract tsfresh features
        features_df = extract_features(
            long_df,
            column_id='id',
            column_sort='time',
            column_value='value',
            impute_function=impute,
            show_warnings=False
        )
        
        # Add company_size and one-hot encode
        features_df["company_size"] = request.company_size
        features_df = pd.get_dummies(features_df, columns=["company_size"])
        
        # Ensure all columns from training are present
        training_columns = model.model.feature_names_in_
        for col in training_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[training_columns]
        
        # Predict
        proba = model.model.predict_proba(features_df)[0][1]
        latency = (time.time() - start_time) * 1000
        
        response = {
            "probability": float(proba),
            "risk": "high" if proba > 0.7 else "medium" if proba > 0.3 else "low",
            "confidence": min(proba * 1.5, 0.99),
            "metrics": model.metrics,
            "warnings": [
                "Trained on historical tsfresh features (1999-2009)",
                "Not validated for financial institutions"
            ],
            "latency_ms": latency,
            "status": "success"
        }
        logger.info(f"Prediction successful: latency={latency:.2f}ms, probability={proba:.4f}")
        return response
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Prediction failed: {str(e)}, latency={latency:.2f}ms")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "latency_ms": latency,
            "status": "error"
        })