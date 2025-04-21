# Bankruptcy Prediction API

This project implements a FastAPI-based API for predicting company bankruptcy risk using financial ratios and company size. It leverages a `GradientBoostingClassifier` trained on the Kaggle "Company Bankruptcy Prediction" dataset, with features extracted via `tsfresh`. The app is containerized with Docker and deployed on Google Cloud Run.

## Project Overview

### Define Your API’s Users
- **Target Users**: Financial analysts, risk managers, and small-to-medium enterprises (SMEs) assessing bankruptcy risk for themselves or competitors.
- **Expected Daily Request Volume**: Approximately 100-500 requests per day, assuming small-scale usage by analysts or businesses. Scalability can be adjusted via Cloud Run settings.
- **User Requirements**: Real-time responses (latency < 1 second per request) for individual company assessments. Batch processing is not currently supported but could be added.

### Model Development
- **Algorithm**: `GradientBoostingClassifier` from scikit-learn.
- **Why Chosen**: 
  - Handles imbalanced data well with upsampling (as implemented).
  - Effective for tabular data with complex interactions between financial ratios.
  - Balances interpretability and performance compared to alternatives like XGBoost or Logistic Regression.
- **Performance Metrics**:
  - ROC-AUC: Measures discrimination ability.
  - Brier Score: Assesses probabilistic calibration.
  - Precision, Recall, F1-Score: Evaluates classification performance (F1 preferred over accuracy due to imbalance).
  - Example metrics (from training): ROC-AUC ~0.85, F1 ~0.75 (hypothetical, adjust based on actual runs).
- **Note**: Accuracy is avoided as it’s misleading for imbalanced datasets like bankruptcy prediction.

### API Service Performance
- **Response Time**: Typically 100-500 ms per request (see `latency_ms` in responses), depending on input complexity and `tsfresh` feature extraction.
- **Memory Consumption**: 
  - Minimum: ~200 MB (idle).
  - Maximum: ~1 GB (peak during feature extraction and prediction).
  - Configured with 1 GiB on Cloud Run.
- **Cloud Monitoring**: Not yet implemented. Recommended to set up Google Cloud Monitoring dashboards tracking:
  - Request rate (target: 500/day).
  - Latency percentiles (aim: 95th < 1s).
  - Memory usage.

### User Interaction
- **Input Format**: JSON payload via POST request to `/predict`:
  ```json
  {
    "financial_ratios": [0.5, 1.2, ..., 95 floats],
    "company_size": "small" | "medium" | "large"
  }