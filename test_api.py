import requests
import pandas as pd
import json
import random
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://bankruptcy-api-xyz.a.run.app"
LOGIN_URL = f"{BASE_URL}/login"
PREDICT_URL = f"{BASE_URL}/predict"
OUTPUT_FILE = "test_results.json"

try:
    DATA_FILE = "data/data.csv"
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        raise FileNotFoundError(f"Please place 'data.csv' in 'data/' directory. Download from Kaggle: 'Company Bankruptcy Prediction'.")
    df = pd.read_csv(DATA_FILE)
    df.rename(columns={'Bankrupt?': 'bankrupt'}, inplace=True)
    df = df.iloc[:, :-1]  # Drop last feature for 94 features
    logger.info("Real data loaded successfully, reduced to 94 features")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    raise

test_cases = [
    {"description": "Linear Increase", "data": {"financial_ratios": [float(i) for i in range(94)]}, "expected": "High probability"},
    {"description": "Constant Low", "data": {"financial_ratios": [0.5] * 94}, "expected": "Low probability"},
    {"description": "Random Values", "data": {"financial_ratios": [random.uniform(0, 100) for _ in range(94)]}, "expected": "Unpredictable"},
    {"description": "Real Bankrupt", "data": {"financial_ratios": df[df['bankrupt'] == 1].iloc[0, :-1].tolist()}, "expected": "High probability"},
    {"description": "Real Non-Bankrupt", "data": {"financial_ratios": df[df['bankrupt'] == 0].iloc[0, :-1].tolist()}, "expected": "Low probability"},
    {"description": "All Zeros", "data": {"financial_ratios": [0.0] * 94}, "expected": "Low probability"},
    {"description": "Invalid Input", "data": {"financial_ratios": [0.0] * 93}, "expected": "422 error"}
]

def get_token():
    response = requests.post(LOGIN_URL, timeout=10)
    if response.status_code == 200:
        token = response.json()["token"]
        logger.info("Obtained JWT token")
        return token
    raise Exception(f"Failed to get token: {response.status_code} {response.text}")

def test_case(case, case_number, token):
    logger.info(f"Running test {case_number}: {case['description']}")
    print(f"\nTest {case_number}: {case['description']}")
    print(f"Expected: {case['expected']}")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, json=case['data'], headers=headers, timeout=10)
    result = {
        "test_number": case_number,
        "description": case['description'],
        "input": case['data'],
        "timestamp": datetime.now().isoformat(),
        "status_code": response.status_code,
        "expected": case['expected']
    }
    
    if response.status_code == 200:
        result["response"] = response.json()
        print(f"Status: {response.status_code} OK")
        print(f"Probability: {result['response']['probability']:.4f}")
        print(f"Risk: {result['response']['risk']}")
    else:
        result["error"] = response.text
        print(f"Status: {response.status_code} Failed")
        print(f"Error: {response.text}")
        if "422" in case["expected"] and response.status_code == 422:
            print("Expected error received")
    
    with open(OUTPUT_FILE, "a") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print("Starting API Tests...")
print("=" * 50)

token = get_token()
print(f"Obtained JWT token: {token[:10]}...")

for i, case in enumerate(test_cases, 1):
    test_case(case, i, token)

print("=" * 50)
print(f"Tests completed. Results saved to {OUTPUT_FILE}")