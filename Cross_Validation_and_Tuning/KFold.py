# Cross validation ka simple matlab:

# > Model ko ek hi data ko multiple angles se test karna
# Taaki overfitting pakda ja sake

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris 
import numpy as np

#  Loading Dataset
iris = load_iris()
X, y = iris.data, iris.target

# model
model=RandomForestClassifier(random_state=42)

# 5-fold Cross Validation
scores = cross_val_score(model, X,y,cv=5)
print("Cross-Validation Scores:",scores)
print("Mean Accuracy:",np.mean(scores))
print("Standard Deviation:",np.std(scores))