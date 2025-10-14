import pandas as pd
# CSV loading
df = pd.read_csv("sample_data.csv")

# Top 5 rows
print(" First 5 rows:")
print(df.head())

# Shape (rows, columns)
print("\n Shape of dataset:", df.shape)

# Columns k names
print("\n Columns:")
print(df.columns)
# Dataset info
print("\n Info:")
print(df.info())

# Missing values check
print("\n Missing values:")
print(df.isnull().sum())

# summary statistics
print("\n Summary statistics:")
print(df.describe())