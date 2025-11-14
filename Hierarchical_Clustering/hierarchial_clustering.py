# HIERARCHICAL CLUSTERING — SUPER SIMPLE EXPLANATION

#  What is Hierarchical Clustering?

# KMeans me hum K (clusters) pehle bata dete the.

# Hierarchical me model khud decide karta hai kaise clusters merge ya break honge.

# Isme tree–like structure use hota hai jise bolte hain:

#  DENDROGRAM

# Ye hamara main tool hoga.


# ---

#  Types of Hierarchical Clustering

# 1️ Agglomerative (Bottom–Up)  we will learn this

# Har point ek individual cluster hota hai

# Dheere-dheere closest clusters merge hote jate

# End me ek bada cluster


# 2️ Divisive (Top–Down)

# – Pehle ek bada cluster
# – Fir break hota hai

# Industry me Agglomerative hi use hota hai.


# ---

#  Why is Hierarchical Useful?

# K-Means jaisa “K choose karo” problem nahi

# Dendrogram se exact K automatically mil jata

# Outliers ko identify karna easy

# Small-medium datasets ke liye best

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Sample dataset (mall customers dataset)
data = {
    "Age": [18,22,25,30,35,40,45,50,60,65],
    "Spending_Score":[80,75,60,65,30,25,20,15,40,10]
}

df = pd.DataFrame(data)
X = df[["Age", "Spending_Score"]]

# Combined Figure
plt.figure(figsize=(12,5))

# ---------------- Graph 1 : Dendrogram ----------------
plt.subplot(1,2,1)
dendrogram(linkage(X, method='ward'))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")

# ---------------- Graph 2 : Clustering ----------------
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df["Cluster"] = model.fit_predict(X)

plt.subplot(1,2,2)
plt.scatter(df["Age"], df["Spending_Score"], c=df["Cluster"], cmap='viridis')
plt.title("Hierarchical Clustering Results")
plt.xlabel("Age")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()


