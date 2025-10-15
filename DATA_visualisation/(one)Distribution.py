import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data2.csv")
# DISTRIBUTION
sns.histplot(df["Marks"], kde=True,bins=10,color="blue")
plt.title("Distribution of marks")
plt.show()
# Bins se na ye hota ki intervals badh jata


