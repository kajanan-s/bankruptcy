import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    
    def fit(self, X: pd.DataFrame):
        if 'company_size' in X.columns:
            self.encoder.fit(X[['company_size']])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'company_size' in X.columns:
            encoded = self.encoder.transform(X[['company_size']]).toarray()
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out()
            )
            X = pd.concat([X.drop('company_size', axis=1), encoded_df], axis=1)
        return X