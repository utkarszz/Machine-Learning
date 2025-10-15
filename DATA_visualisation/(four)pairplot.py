import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data2.csv")
sns.pairplot(df, hue="Grade",diag_kind="kde")
plt.show()