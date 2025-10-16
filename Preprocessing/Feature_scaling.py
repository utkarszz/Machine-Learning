import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
  "Marks": [50,60,70,80,90],
  "Age": [18,20,22,24,26]
}

df = pd.DataFrame(data)
print("\nOriginal Data :\n", df)

# Standarization (Z-score Scaling)[Iska formula tha usse nikala ans]
scaler = StandardScaler()
df_standard = scaler.fit_transform(df)
print("\nStandardization:\n",df_standard)
#  Normalization(Min-Max Scaling)[Iska formula tha usse nikala ans]
scaler= MinMaxScaler()
df_normal = scaler.fit_transform(df)
print("\nNormalization:\n", df_normal)