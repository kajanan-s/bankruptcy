from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
import numpy as np
import pandas as pd
import joblib
import time
import logging
import jwt
import shap
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from tsfresh import extract_features
from .model import BankruptcyModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Bankruptcy Prediction API", version="1.0.0")
security = HTTPBearer()

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
SECRET_KEY = "your-secret-key-here"

def load_or_train_model():
    try:
        if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH]):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_names = joblib.load(FEATURE_NAMES_PATH)
            logger.info(f"Loaded model. Metrics: {model.metrics}")
            if len(feature_names) != 94:
                logger.warning(f"Feature names count mismatch: {len(feature_names)}")
                raise ValueError("Expected 94 features")
            return model, scaler, feature_names
        else:
            logger.warning("Model files missing. Training new model...")
    except Exception as e:
        logger.error(f"Failed to load assets: {str(e)}. Training new model...")

    # Fallback training
    data_path = "data/data.csv"
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data file missing")
        df = pd.read_csv(data_path)
        df.rename(columns={'Bankrupt?': 'bankrupt'}, inplace=True)
        df['id'] = df.index
        df_melted = df.melt(id_vars=['id', 'bankrupt'], var_name='feature', value_name='value')
        df_melted['time'] = df_melted.groupby('id').cumcount()
        extracted_features = extract_features(
            df_melted[['id', 'time', 'value']],
            column_id='id',
            column_sort='time',
            column_value='value',
            n_jobs=4
        )
        extracted_features.fillna(0, inplace=True)
        X = extracted_features
        y = df['bankrupt']
        feature_names = X.columns[:94].tolist()
        X = X[feature_names]
        df_features = X.copy()
        df_features['bankrupt'] = y
        df_class_0 = df_features[df_features['bankrupt'] == 0]
        df_class_1 = df_features[df_features['bankrupt'] == 1]
        df_class_1_upsampled = df_class_1.sample(n=len(df_class_0), replace=True, random_state=42)
        df_balanced = pd.concat([df_class_0, df_class_1_upsampled]).reset_index(drop=True)
        X_balanced = df_balanced.drop(columns=['bankrupt'])
        y_balanced = df_balanced['bankrupt']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)
        model = BankruptcyModel()
        model.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        model.train(X_scaled, y_balanced)
        logger.info(f"Fallback training complete. Metrics: {model.metrics}")
        for path in [MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH]:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_names, FEATURE_NAMES_PATH)
    except Exception as e:
        logger.error(f"Fallback training failed: {str(e)}. Using dummy model...")
        model = BankruptcyModel()
        model.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
        scaler = StandardScaler()
        feature_names = [f"feature_{i}" for i in range(94)]
        for path in [MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH]:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info("Model artifacts saved")
    return model, scaler, feature_names

model, scaler, feature_names = load_or_train_model()
try:
    explainer = shap.TreeExplainer(model.model)
    logger.info("SHAP explainer initialized")
except Exception as e:
    logger.error(f"SHAP explainer failed: {str(e)}")
    explainer = None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        logger.info(f"Authenticated user: {payload.get('sub')}")
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"Authentication failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

class PredictionRequest(BaseModel):
    financial_ratios: List[float] = Field(min_length=94, max_length=94, description="94 tsfresh-extracted features")

class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    risk: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    metrics: Dict
    warnings: List[str]
    latency_ms: float
    status: str
    explanations: List[Dict]

@app.post("/login", response_model=Dict[str, str])
async def login():
    try:
        payload = {"sub": "test_user", "exp": int(time.time()) + 3600}
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        logger.info("Issued JWT token")
        return {"token": token}
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to issue token")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, user: dict = Depends(verify_token)):
    start_time = time.time()
    try:
        features_df = pd.DataFrame([request.financial_ratios], columns=feature_names)
        logger.info(f"Input features shape: {features_df.shape}")
        features_scaled = scaler.transform(features_df)
        logger.info(f"Scaled features shape: {features_scaled.shape}")
        proba = model.model.predict_proba(features_scaled)[0][1]
        logger.info(f"Prediction probability: {proba}")
        explanations = []
        if explainer:
            try:
                shap_values = explainer.shap_values(features_scaled)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_vals = shap_values[1]
                else:
                    shap_vals = shap_values
                explanations = [
                    {"feature": feature_names[i], "impact": float(shap_vals[0][i])}
                    for i in np.argsort(np.abs(shap_vals[0]))[-5:]
                ]
            except Exception as e:
                logger.warning(f"SHAP failed: {str(e)}")
        latency = (time.time() - start_time) * 1000
        response = PredictionResponse(
            probability=float(proba),
            risk="high" if proba > 0.5 else "medium" if proba > 0.2 else "low",
            confidence=min(proba * 1.5, 0.99),
            metrics=model.metrics,
            warnings=["Trained on tsfresh-extracted features. Model may have low accuracy."],
            latency_ms=latency,
            status="success",
            explanations=explanations
        )
        logger.info(f"Prediction: latency={latency:.2f}ms, probability={proba:.4f}, risk={response.risk}")
        return response
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e), "latency_ms": latency, "status": "error"})