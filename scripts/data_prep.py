import pandas as pd
from sklearn.datasets import fetch_openml
import os

def prepare_data():
    # Load dataset
    data = fetch_openml("company-bankruptcy-prediction", version=1, as_frame=True)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['bankrupt'] = data.target
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    df.to_csv("data/bankruptcy_data.csv", index=False)
    print("Dataset prepared and saved to data/bankruptcy_data.csv")

if __name__ == "__main__":
    prepare_data()