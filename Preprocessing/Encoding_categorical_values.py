from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
data = {
  "Name": ["A","B","C","D","E"],
  "City":["Delhi","Mumbai","Delhi","Chennai","Mumbai"]
}
df = pd.DataFrame(data)

# Label Encoding
le = LabelEncoder()
df["City_Label"] = le.fit_transform(df["City"])
print("\nLabel Encoding :\n", df)

# One hot encoding
df_onehot = pd.get_dummies(df,columns =["City"])
print("\nOne Hot Encoding:\n", df_onehot)

