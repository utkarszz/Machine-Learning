import pandas as pd
# CSV loading
df = pd.read_csv("sample_data.csv")
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()