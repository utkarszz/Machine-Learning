# PCA (Principal Component Analysis) — Complete Guide

# PCA ka main purpose:

# “Data ko compress karna WITHOUT losing much information.”

# Jaise:

# 30 features → 2 features

# 10 features → 3 features

# 4 features → 2 features (visualization ke liye)


# Machine Learning me PCA use hota hai:
# ✔ Visualization
# ✔ Noise removal
# ✔ Fast model training
# ✔ Overfitting kam karne ke liye
# ✔ Feature correlation resolve karne ke liye


# ---

#  STEP 1 — Why PCA? (Real intuition)

# Data ke columns kabhi-kabhi:

# Duplicate information rakhte hain

# Highly correlated hote hain

# Useless noise hote hain


# Example:

# Height (cm)

# Height (inches)

# Height (feet)


# Ye 3 features different lag rahe… but all convey the SAME information.

# PCA ye karta hai:

# Sare correlated features ko combine karke ek new feature banata hai

# Jitna important information hoga, utna variance capture karega



# ---

#  STEP 2 — PCA ka simple idea (BINA maths)

# Socho ek scatter plot hai:

# *
#     *          *
#         *  *
#       *

# Data diagonal direction me faila hua hai.
# But hamne features x-axis, y-axis rakh diya.

# Real information diagonal direction me hai → PCA axis rotate karta hai us direction me.

# Ye axis = Principal Components (PC1, PC2).

# PC1 = sabse important information

# PC2 = bachi hui information


# PC1 always > PC2 in importance.


# ---

#  STEP 3 — Kya input chahiye PCA ke liye?

# Numerical data

# Scaling required (VERY IMPORTANT)


# PCA scaling ke bina bekaar hota hai.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Loading Dataset
iris=load_iris()
X=iris.data
y=iris.target
# Standarizing the data
scaler =StandardScaler()
X_scaled=scaler.fit_transform(X)

# Apply PCA{we want 2 components for visualization}
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape :" ,X.shape)
print("Transformed Shape:",X_pca.shape)

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=y, cmap="viridis")
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.title("PCA Visualization of Iris Dataset")
plt.colorbar(label="Classes")
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", sum(pca.explained_variance_ratio_))

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

#  STEP 8 — PCA ka kab use nahi karna?

#  If original features already very meaningful
#  If data categorical ho
#  If interpretability important hai
#  If target variable numeric relationships strong hain
#  Deep learning me PCA ki zarurat nahi


# ---

#  STEP 9 — YOUR PCA SUMMARY (Very Clean – Useful for Notes)

# PCA (Principal Component Analysis)

# PCA ek dimensionality reduction technique hai

# Jo correlated features ko combine kar deta hai

# Data ke axis ko rotate karta hai taaki maximum variance capture ho

# PC1 > PC2 in importance

# Scaling zaroori hai

# Used for visualization, noise removal, faster training

# Also used before KMeans/SVM/KNN
