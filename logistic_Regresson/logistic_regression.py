# Concept: Logistic Regression

# Naam me “Regression” hai, but ye Classification algorithm hai.

# Jab output categorical ho (Yes/No, Pass/Fail, Spam/Not Spam) tab use hota hai.

# Linear Regression ek straight line banata hai,

# Logistic Regression ek S-shaped (sigmoid) curve banata hai jiska output 0 aur 1 ke beech hota hai.

# Agar output > 0.5 → Class = 1

# Agar output ≤ 0.5 → Class = 0

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Dataset
data = {
  "Hours_Studied":
  [1,2,3,4,5,6,7,8,9,10],
  "Pass":[0,0,0,0,0,1,1,1,1,1]
  # fail = 0 & pass = 1
}
df = pd.DataFrame(data)

# Features & Target
X = df[["Hours_Studied"]]
Y = df["Pass"]

# Test train split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train,Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
print("Accuracy :",accuracy_score(Y_test,Y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(Y_test,Y_pred))
print("\nClassification Report:\n",classification_report(Y_test,Y_pred))

# scatter plot of actual data
plt.scatter(df["Hours_Studied"], df["Pass"], color = "blue", label ="Actual Data")

# generate a smooth range of hrs for prediction
X_range = np.linspace(0, 10, 100).reshape(-1,1)
Y_prob = model.predict_proba(X_range)[:,1] #probability of pass = 1

# Plot logistic regression curve(S-shaped sigmoid)

plt.plot(X_range,Y_prob, color = "red", label = "Logistic Regression Curve")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Pass/fail Prediction using Logistic Regression")
plt.legend()
plt.show()