import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
}

df = pd.DataFrame(data)
print(df)
print(df.describe())
print(df['Name'])
print(df[df['Age'] > 28])
print(df.head())  # First 5 rows
print(df.tail())  # Last 5 rows
print(df["Age"].mean())  # Average age