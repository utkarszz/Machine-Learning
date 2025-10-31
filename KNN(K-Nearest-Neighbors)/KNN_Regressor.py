# Concept in Simple Words:

# Classification me KNN majority vote se decide karta tha (Yes/No).

# Regression me KNN average nikalta hai nearby points ka (numerical prediction).


# Matlab:

# > Agar kisi student ne 8 hrs study ki, to uske aas paas (jaise 7, 9, 10 hrs)
# students ke marks ka average lekar predict karega — ye student kitne marks laayega.

# KNN Regressor:-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset
data = {
  "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
  "Sleep_Hours":[8,7,6,6,5,5,4,3,3,2],
  "Marks":[20,25,30,45,50,60,65,75,85,95]
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Hours_Studied", "Sleep_Hours"]]
y = df["Marks"]

#  Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = KNeighborsRegressor(n_neighbors=3)  # k=3
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("mean_squared_error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Comparing actual vs predicted
result = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print("\nComparison of Actual vs Predicted:\n", result)

# Output Explanation

# Mean Squared Error (MSE):
# Measures how far predictions are from actual marks
# → Lower MSE = better model

# R² Score:
# Tells how well model fits the data
# → Closer to 1 means excellent accuracy

# Prediction Table:
# Compares real marks vs model’s predicted marks.

# Visualization:
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("KNN Regression - Actual vs Predicted Marks")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()