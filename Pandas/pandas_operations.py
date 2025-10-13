#select column
import pandas as pd

# Dataframe
data = {
  "Name": ["Alice", "Bob", "Charlie", "David"],
  "Marks": [85, 90, 78, 92],
  "Passed": [True, True, False, True]

}
df = pd.DataFrame(data)
print(df["Marks"])
#Filter rows
print(df[df["Marks"]>80])

# Add new column
df["Grade"] = ["A", "A", "B", "A"]

# Drop column
df = df.drop("Passed", axis=1)
print(df)