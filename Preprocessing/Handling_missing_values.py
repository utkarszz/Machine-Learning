import pandas as pd
#  making sample data
data = {
  "Name":["A","B","C","D","E"],
  "Age" : [20,25,None,30,22],
  "Marks":[85,None,75,90,88]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Missing values count
print("\nMissing Values :\n", df.isnull().sum())

# Drop rows with missing values
df_drop = df.dropna()
print("\nAfter Dropping :\n", df_drop)

# Fill missing values(Mean Manipulation)
df_fill = df.fillna(df.mean(numeric_only=True))
print("\nAfter Filling :\n", df_fill)