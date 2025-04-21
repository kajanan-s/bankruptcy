Bankruptcy Prediction API
This project is a bankruptcy prediction API built for Assignment 2. It uses a GradientBoostingClassifier trained on 94 tsfresh features extracted from data/data.csv to predict bankruptcy risk. The API, built with FastAPI, provides a /predict endpoint for predictions and a /login endpoint for JWT authentication. It is deployed using Docker and Google Cloud Run, meeting test requirements: 200 OK for Tests 1-6 and 422 for Test 7 in test_api.py.
Features

FastAPI Backend: Serves predictions (/predict) and authentication (/login) on port 8080.
Model: GradientBoostingClassifier with tsfresh features for robust bankruptcy prediction.
Testing: test_api.py validates API behavior (e.g., low/high probabilities, input errors).
Deployment: Runs locally with Docker or on Google Cloud Run.

Prerequisites

Python: 3.9
Docker: For containerized deployment
Google Cloud SDK: For Cloud Run deployment
Dependencies: Listed in requirements.txt

Project Structure
bankruptcy/
├── app/
│   └── main.py             # FastAPI backend
├── data/
│   └── data.csv           # Dataset for feature extraction
├── test_api.py            # Test script
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── model.pkl              # Trained model
├── scaler.pkl             # Feature scaler
├── feature_names.pkl      # Feature names
└── README.md              # This file

Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/bankruptcy.git
cd bankruptcy

2. Install Dependencies
Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

3. Verify Artifacts
Ensure model.pkl, scaler.pkl, feature_names.pkl, and data/data.csv are present. To verify the model:
python -c "import joblib; print(joblib.load('model.pkl').metrics)"

Expected output: {'accuracy': 0.9418, 'precision': 0.9004, 'recall': 0.9935, 'f1': 0.9447}
4. Run Locally (Without Docker)
Run the FastAPI server:
uvicorn app.main:app --host 0.0.0.0 --port 8080

Access at http://localhost:8080/docs.
5. Run with Docker
Build and run the Docker container:
docker build -t bankruptcy-api .
docker run -p 8080:8080 bankruptcy-api

Access at http://localhost:8080/docs.
Usage

Test API Locally:

Run test_api.py to verify functionality:
python test_api.py

Expected output:
2025-04-20 ... - INFO - Real data loaded successfully, reduced to 94 features
Starting API Tests...
Test 1: Linear Increase - Status: 200 OK, Probability: 0.0154, Risk: low
...
Test 7: Invalid Input - Status: 422 Failed, Error: {"detail":[{"type":"too_short",...}]}




Manual Testing:

Get a JWT token:
curl -X POST http://localhost:8080/login


Predict with 94 financial ratios:
curl -X POST http://localhost:8080/predict \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"financial_ratios": [0.0, 0.0, ..., 0.0]}'





Cloud Run Deployment
1. Authenticate
gcloud auth login
gcloud config set project insys-457502
gcloud auth configure-docker

2. Build and Push Image
docker build -t bankruptcy-api .
docker tag bankruptcy-api gcr.io/insys-457502/bankruptcy-api:latest
docker push gcr.io/insys-457502/bankruptcy-api:latest

3. Deploy to Cloud Run
gcloud run deploy bankruptcy-api \
  --image gcr.io/insys-457502/bankruptcy-api:latest \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --memory 512Mi \
  --allow-unauthenticated

Access at the provided Cloud Run URL (e.g., https://bankruptcy-api-xyz.a.run.app).
4. Test on Cloud Run
Update BASE_URL in test_api.py to the Cloud Run URL:
BASE_URL = "https://bankruptcy-api-xyz.a.run.app"

Run:
python test_api.py

Troubleshooting

PowerShell Errors: If gcloud fails:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass -Force


Docker Push Fails: Ensure gcloud auth configure-docker is run.

Cloud Run 404 Error: Check logs:
gcloud run services logs read bankruptcy-api --region us-central1

Verify port 8080 and endpoint availability in main.py.

Model Issues: If model.pkl fails, retrain:
python train_model.py
docker build -t bankruptcy-api .


Red model.pkl in Git: Likely untracked. Add to .gitignore:
echo "*.pkl" >> .gitignore
git add .gitignore
git commit -m "Ignore pickle files"



