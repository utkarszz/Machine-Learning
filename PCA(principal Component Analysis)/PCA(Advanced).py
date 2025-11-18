# 1️ Explained Variance Ratio (MOST IMPORTANT)

# PCA dimensionality drastically reduce karta hai.
# Par kaunse components useful hai? Usko find karne ke liye:

#  Explained Variance Ratio batata hai:
# PC1 kitni information carry karta h
# PC2 kitni karta h
# PC3 kitni karta h, etc
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn  import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading  dataset
iris = datasets.load_iris()
X=iris.data

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
pca.fit(X_scaled)

print("Explained Varience Ratio:",pca.explained_variance_ratio_)
print("Total Variance Covered:", sum(pca.explained_variance_ratio_))

# Scree Plot
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Scree Plot-PCA")
plt.grid(True)
plt.show()
# Graph Explanation:

# Jaha graph flat hona start ho jaye →
# Waha ke baad components useless hai.


# Usually iris me:

# PC1 + PC2 → 95% info keep
# PC3 + PC4 → almost useless

# Logistic Regression(PCA+Classification)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# keeping only 2 components
pca= PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train test Split
X_train,X_test,y_train,y_test = train_test_split(X_pca, iris.target,test_size=0.2,random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train,y_train)

# Prediction
y_pred= model.predict(X_test)
print("Accuracy with PCA:" , accuracy_score(y_test,y_pred))


# PCA+KMeans Clustering
from sklearn.cluster import KMeans

kmeans= KMeans(n_clusters=3,random_state=42)
labels =kmeans.fit_predict(X_pca)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1],c=labels,cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA+KMeans Clustering")
plt.show()