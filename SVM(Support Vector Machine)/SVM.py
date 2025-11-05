# Why SVM is important?

# SVM is used in:

# Face recognition

# Cancer detection

# Fraud detection

# Handwriting recognition

# Hyperplane based decision making


# SVM = Powerful classifier
# Ye data ko separate karta hai best boundary (hyperplane) se —
# jisme maximum margin hota hai.


# ---

#  SVM Concepts You Will Learn:

# Concept	Meaning

# Hyperplane	Line/Boundary dividing classes
# Margin	Distance between hyperplane & nearest points
# Support Vectors	Border pe jo points hote hain — decision makers
# Kernels	Non-linear data ko linear banane ka trick


# Kernel types:

# Linear

# Polynomial

# RBF (Radial Basis Function)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using first two features for easy visualization
Y = iris.target

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model
model = SVC(kernel="rbf")  # Radial Basis Function kernel linear and poly bhi hota h
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# Visualization
x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx , yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)

# Scatter plot of actual data points
plt.scatter(X[:, 0], X[:, 1], c=Y,  cmap="coolwarm", s = 80)

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200, facecolors='none', edgecolors='black',linewidth=2, label='Support Vectors')

plt.title("SVM Decision Boundary")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()