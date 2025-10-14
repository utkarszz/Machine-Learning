import pandas as pd
# CSV loading
df = pd.read_csv("sample_data.csv")
# Fill missing values with mean
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
# nothing to print