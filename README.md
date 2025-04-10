# Bankruptcy Prediction API
python version 3.16.6
## Setup Locally
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

# Activate virtual environment (if not already active)
source venv/bin/activate  # Windows: venv\Scripts\activate

# Prepare data
python prepare_data.py

# Train model
python train_model.py

# Run API
uvicorn app.main:app --reload

Model trained with metrics: {'roc_auc': 0.9910003581809916, 'brier_score': 0.008364318392847691, 'precision': 1.0, 'recall': 0.6818181818181818, 'f1_score': 0.8108108108108109}

# Bankruptcy Prediction API

## Setup
1. Clone: `git clone <repo-url>`
2. Install: `pip install -r requirements.txt`
3. Download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) and place in `data/`.
4. Prepare data: `python prepare_data.py`
5. Train model: `python train_model.py`
6. Run API: `uvicorn app.main:app --reload`
7. Test: `python test_api.py`

## Users
- **Target**: Financial analysts, small business owners.
- **Volume**: ~100 requests/day.
- **Requirements**: Real-time JSON responses (<5s).

## Diagram
[User] -> [POST /predict] -> [API] -> [tsfresh + Model] -> [JSON Response]

## Deployment
1. Build Docker image:
   ```bash
   docker build -t bankruptcy-api:latest .