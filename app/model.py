import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from .data_preprocessing import DataPreprocessor

class BankruptcyModel:
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.metrics = {}
    
    def train(self, data_path: str):
        df = pd.read_csv(data_path)
        X = df.drop('bankrupt', axis=1)
        y = df['bankrupt']
        
        self.preprocessor.fit(X)
        X_processed = self.preprocessor.transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.model.fit(X_processed, y)
        
        # Store metrics
        self.metrics = self._evaluate(X_processed, y)
        
        joblib.dump(self, "model.pkl")
    
    def _evaluate(self, X, y):
        probas = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)
        return {
            'roc_auc': float(roc_auc_score(y, probas)),
            'brier_score': float(brier_score_loss(y, probas))
        }

def load_model(model_path: str = "model.pkl"):
    return joblib.load(model_path)