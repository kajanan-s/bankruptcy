import pandas as pd
import joblib
import logging
from sklearn.utils import resample
from app.model import BankruptcyModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    df = pd.read_csv("data/bankruptcy_data_tsfresh.csv")
    
    # Separate majority and minority classes
    df_majority = df[df['bankrupt'] == 0]
    df_minority = df[df['bankrupt'] == 1]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    X = df_balanced.drop(columns=["bankrupt"])
    X = pd.get_dummies(X, columns=["company_size"])
    y = df_balanced["bankrupt"]
    
    model = BankruptcyModel()
    model.train(X, y)
    joblib.dump(model, "model.pkl")
    logger.info("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()