# Decision Tree ek supervised learning algorithm hai jo classification aur regression dono ke liye use hota hai.
# Ye data ko tree ke format me divide karta hai — har node ek feature represent karta hai, aur leaf node ek final decision/output.

# Example Use Cases:

# Spam detection (spam / not spam)

# Loan approval (approve / reject)

# Medical diagnosis (disease / no disease)

# Weather prediction (rain / no rain)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
  "Age":
  [22,25,47,52,46,56,55,60,30,28],
  "Salary":
  [25000,40000,45000,60000,35000,80000,70000,120000,500000,300000],
  "Buys_Car":[0,0,1,1,0,1,1,1,1,0]
  # 1 = buys car, 0 =no
}
df = pd.DataFrame(data)

# Features and target
X = df[["Age","Salary"]]
Y = df["Buys_Car"]

# Split data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,random_state=42)

# Model
model = DecisionTreeClassifier(criterion="entropy", random_state= 42)
model.fit(X_train,Y_train)

# prediction
Y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:",accuracy_score(Y_test,Y_pred))
print("\nConfusion Matrix :\n",confusion_matrix(Y_test,Y_pred))
print("\nClassification Report :\n", classification_report(Y_test,Y_pred))

# Visualization
plt.figure(figsize=(10,6))
tree.plot_tree(model,feature_names=["Age", "Salary"],class_names=["No","Yes"],filled=True)
plt.show()