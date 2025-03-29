from app.model import BankruptcyModel
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def train_and_save_model():
    model = BankruptcyModel()
    model.train("data/bankruptcy_data.csv")
    print(f"Model trained with metrics: {model.metrics}")

if __name__ == "__main__":
    train_and_save_model()