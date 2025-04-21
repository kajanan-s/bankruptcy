#ooga
import pandas as pd
df = pd.read_csv("data/data.csv")
print("Shape:", df.shape)  # Should be (6819, 96)
print("Columns:", df.columns.tolist())  # Should include 'Bankrupt?' and 95 features
print("Bankrupt? value counts:\n", df['Bankrupt?'].value_counts())  # ~6599 0s, ~220 1s
print("Missing values:", df.isnull().sum().sum())  # Should be 0 or small
print("Sample data:\n", df.head()) 