import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data2.csv")
# correlation heatmap mei sirf numeric values select hote
numeric_df=df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr() , annot=True,cmap="coolwarm")
plt.title("correlation Heatmap")
plt.show()