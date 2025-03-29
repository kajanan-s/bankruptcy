from fastapi.testclient import TestClient
from app.main import app
import numpy as np

client = TestClient(app)

def test_predict():
    test_data = {
        "financial_ratios": list(np.linspace(0, 100, 64)),
        "company_size": "medium"
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probability"] <= 1
    assert data["risk"] in ["low", "medium", "high"]