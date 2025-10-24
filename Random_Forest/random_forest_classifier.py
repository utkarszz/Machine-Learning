# 🔹 1️⃣ What is Random Forest?

# Random Forest ek Ensemble Learning Algorithm hai —
# matlab ye multiple Decision Trees banata hai, aur un sabka result combine karke final output deta hai.

# Soch le — ek single Decision Tree jaise ek individual teacher hai.
# Lekin Random Forest me 100 teachers (trees) milke decision lete hain →
# majority jo bole, wahi final output hota hai ✅


# ---

# 🔹 2️⃣ Why Random Forest?

# Single Decision Tree:

# Easily overfit karta hai (yaani training data me perfect, test data me weak)

# Slightly biased ho sakta hai


# Random Forest:

# Multiple trees ka average ya voting use karta hai

# Overfitting kam hota hai

# Accuracy high hoti hai



# ---

# 🔹 3️⃣ Algorithm Intuition

# For classification problems:

# 1. Dataset ka random subset liya jaata hai


# 2. Random features select karke alag-alag Decision Trees banaye jaate hain


# 3. Har tree apna prediction deta hai


# 4. Final output = majority vote

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

# Sample Dataset
data = {
  "Age":
  [22,25,47,52,46,56,55,60,30,28],
  "Salary":
  [25000, 40000, 45000, 60000,35000,80000,70000,120000,50000,30000],
  "Buys_Car":[0,0,1,1,0,1,1,1,1,0]
  # 1 = buys car, 0 = no

}
df = pd.DataFrame(data)

# Feature & target
X = df[["Age","Salary"]]
Y = df["Buys_Car"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:",accuracy_score(Y_test, Y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(Y_test,Y_pred))
print("\nClassification Report:\n",classification_report(Y_test,Y_pred))

# Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

feat_importances = pd.Series(model.feature_importances_,index=X.columns)
sns.barplot(x=feat_importances,y=feat_importances.index)
plt.title("Feature Importance")
plt.show()