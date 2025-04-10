import requests
import pandas as pd
import json
import random
import os
from datetime import datetime
import time

# API endpoint
URL = "http://localhost:8000/predict"

# Output file for results
OUTPUT_FILE = "test_results.json"

# Ensure data.csv exists
DATA_FILE = "data/data.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Please place 'data.csv' in 'data/' directory. Download from Kaggle: 'Company Bankruptcy Prediction'.")

# Load real data
df = pd.read_csv(DATA_FILE)
df.rename(columns={'Bankrupt?': 'bankrupt'}, inplace=True)

# Test cases (unchanged)
test_cases = [
    {"description": "Linear Increase (Previous Test)", "data": {"financial_ratios": [float(i) for i in range(95)], "company_size": "medium"}, "expected": "High probability due to strong trend"},
    {"description": "Constant Low Values", "data": {"financial_ratios": [0.5] * 95, "company_size": "small"}, "expected": "Low probability due to stability"},
    {"description": "Cyclic Pattern", "data": {"financial_ratios": [float(i % 10) for i in range(95)], "company_size": "large"}, "expected": "Moderate probability due to oscillation"},
    {"description": "Random Values", "data": {"financial_ratios": [random.uniform(0, 100) for _ in range(95)], "company_size": "medium"}, "expected": "Unpredictable; depends on random pattern"},
    {"description": "Real Data - Bankrupt (First Instance)", "data": {"financial_ratios": df[df['bankrupt'] == 1].iloc[0, :-1].tolist(), "company_size": "medium"}, "expected": "High probability (known bankrupt)"},
    {"description": "Real Data - Non-Bankrupt (First Instance)", "data": {"financial_ratios": df[df['bankrupt'] == 0].iloc[0, :-1].tolist(), "company_size": "large"}, "expected": "Low probability (known non-bankrupt)"},
    {"description": "Edge Case - All Zeros", "data": {"financial_ratios": [0.0] * 95, "company_size": "small"}, "expected": "Low probability or error if extreme"},
    {"description": "Edge Case - All Max", "data": {"financial_ratios": [100.0] * 95, "company_size": "large"}, "expected": "High probability or error if extreme"}
]

invalid_case = {"description": "Invalid Input - Too Few Ratios", "data": {"financial_ratios": [0.0] * 94, "company_size": "medium"}, "expected": "Should return 422 error"}

# Function to test a single case with retry
def test_case(case, case_number, max_retries=3):
    print(f"\nTest {case_number}: {case['description']}")
    print(f"Expected: {case['expected']}")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(URL, json=case['data'], timeout=10)
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
                print(f"Latency: {result['response']['latency_ms']:.2f}ms")
            else:
                result["error"] = response.text
                print(f"Status: {response.status_code} Failed")
                print(f"Error: {response.text}")
            
            with open(OUTPUT_FILE, "a") as f:
                json.dump(result, f)
                f.write("\n")
            return  # Success, exit retry loop
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                print(f"Error after {max_retries} attempts: {str(e)}")
                with open(OUTPUT_FILE, "a") as f:
                    json.dump({
                        "test_number": case_number,
                        "description": case['description'],
                        "input": case['data'],
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "expected": case['expected']
                    }, f)
                    f.write("\n")

# Clear previous results
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Run all tests
print("Starting API Tests...")
print("=" * 50)

for i, case in enumerate(test_cases + [invalid_case], 1):
    test_case(case, i)

print("=" * 50)
print(f"Tests completed. Results saved to {OUTPUT_FILE}")