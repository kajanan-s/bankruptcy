import pandas as pd
import os
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

def prepare_data():
    try:
        input_file = "data/data.csv"
        output_file = "data/bankruptcy_data_tsfresh.csv"
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Please download 'data.csv' from Kaggle and place it in 'data/' directory.")
        
        # Load the dataset
        df = pd.read_csv(input_file)
        
        # Rename target column
        df.rename(columns={'Bankrupt?': 'bankrupt'}, inplace=True)
        
        # Simulate time-series: treat 95 financial ratios as time steps
        ratio_columns = [col for col in df.columns if col not in ['bankrupt']]
        long_df = pd.melt(
            df.reset_index(),
            id_vars=['index'],
            value_vars=ratio_columns,
            var_name='time',
            value_name='value'
        )
        long_df.rename(columns={'index': 'id'}, inplace=True)
        
        # Extract features with tsfresh
        extracted_features = extract_features(
            long_df,
            column_id='id',
            column_sort='time',
            column_value='value',
            impute_function=impute,  # Handle NaNs
            show_warnings=False
        )
        
        # Add target column back
        extracted_features['bankrupt'] = df['bankrupt']
        
        # Add company_size (simulated from a financial metric)
        extracted_features['company_size'] = pd.qcut(df[' Net Income to Total Assets'], q=3, labels=['small', 'medium', 'large'])
        
        # Save to CSV
        os.makedirs("data", exist_ok=True)
        extracted_features.to_csv(output_file, index=False)
        print(f"Dataset with tsfresh features prepared and saved to {output_file}")
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    prepare_data()