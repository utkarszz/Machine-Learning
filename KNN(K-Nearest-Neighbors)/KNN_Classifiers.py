# 1. Understanding the Concept:

# KNN = K-Nearest Neighbors
# It’s a supervised learning algorithm used for both classification and regression,
# but mostly for classification.

# Think of it like this 👇

# > Jab koi naya data point aata hai, to model dekhta hai —
# “Iske sabse kareeb wale (nearest) k data points kaun se hain?”

# Fir unke majority vote ke basis pe decide karta hai ki naya point kis class ka hai.




# ---

# 🧩 Example:

# Imagine tu exam me student ke marks aur study hours ka data le raha hai:

# Hours_Studied	Marks	Result

# 2	30	Fail
# 4	50	Pass
# 5	65	Pass
# 1	25	Fail


# Ab agar ek naya student hai jiska Hours_Studied = 3,
# to KNN dekhega uske 3 nearest points (k=3)
# aur unme se majority kya hai.

# 👉 Agar 2 “Pass” aur 1 “Fail” mila,
# to result ho gaya — “Pass”


# ---

# 📏 2. Distance Formula (How “Nearest” is Calculated):

# Usually Euclidean Distance use hoti hai:

# Distance = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}

# Matlab jitna kam distance, utna zyada similarity.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample Dataset
data = {
  "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
  "Sleep_Hours":[8,7,6,6,5,5,4,3,3,2],
  "Pass":[0,0,0,1,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

# Features and Target
X =df[["Hours_Studied", "Sleep_Hours"]]
y = df["Pass"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = KNeighborsClassifier(n_neighbors=3)  # k=3
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Output Explanation:

# Accuracy: Shows how many predictions were correct

# Confusion Matrix: Tells how many Pass/Fail predictions were correct or incorrect

# Classification Report: Gives precision, recall, and F1 score


# n_neighbors=3 means model looks at 3 nearest data points to decide.

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.scatterplot(x =df["Hours_Studied"], y =df["Sleep_Hours"], hue=df["Pass"],s=100)
plt.title("KNN Classification - Pass/Fail by Study & Sleep Hours")
plt.xlabel("Hours Studied")
plt.ylabel("Sleep Hours")
plt.show()