import pandas as pd
# CSV loading
df = pd.read_csv("sample_data.csv")
print(df.groupby("Grade")["Marks"].mean())