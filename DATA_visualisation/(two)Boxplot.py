import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data2.csv")
sns.boxplot(x=df["Marks"])
plt.title("Boxplot of marks")
plt.show()