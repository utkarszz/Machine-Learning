import pandas as pd

# Dataframe
data = {
  "Name": ["Alice", "Bob", "Charlie", "David"],
  "Marks": [85, 90, 78, 92],
  "Passed": [True, True, False, True]

}
df = pd.DataFrame(data)
print(df.head())  #first rows
print(df.info())  #structure
print(df.describe()) #stats summary