import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset(mall customers dataset)
data = {
    "Age": [18,22,25,30,35,40,45,50,60,65],
    "Spending_Score":[80,75,60,65,30,25,20,15,40,10]
}

df = pd.DataFrame(data)

# Selecting features
X = df[["Age", "Spending_Score"]]

# KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print(df)

# Visualization
plt.scatter(df["Age"],
df["Spending_Score"], c=df["Cluster"], cmap='viridis')
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using KMeans Clustering")
plt.show()