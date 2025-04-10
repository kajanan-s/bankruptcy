from sklearn.ensemble import GradientBoostingClassifier

class BankruptcyModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.metrics = {}

    def train(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score
        import logging

        logger = logging.getLogger(__name__)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "brier_score": brier_score_loss(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        logger.info(f"Model trained with metrics: {self.metrics}")