import pandas as pd
# CSV loading
df = pd.read_csv("sample_data.csv")
print(df.sort_values("Marks",ascending= False))