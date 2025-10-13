# Load CSV file
import pandas as pd
df = pd.read_csv("sample_data.csv")
print(df.shape) # (rows, columns)
print(df.columns) # column names
print(df.isnull().sum()) # missing values per column

#  Sort by column
print(df.sort_values("Marks", ascending=False))

# Group by
print(df.groupby("Grade")["Marks"].mean())
      