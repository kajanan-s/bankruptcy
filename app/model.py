from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankruptcyModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.metrics = {}

    def train(self, X, y):
        try:
            logger.info("Starting model training...")
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            self.metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred)),
                "recall": float(recall_score(y, y_pred)),
                "f1": float(f1_score(y, y_pred))
            }
            logger.info(f"Training completed. Metrics: {self.metrics}")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
            raise