# 🧠 Concept: What is Random Forest Regressor?

# Random Forest Regressor ek ensemble model hai jisme multiple Decision Tree Regressors banaye jaate hain —
# aur har ek tree apna prediction deta hai.

# Finally, model sab trees ke average ko final output ke roop me deta hai.
# Yani har tree apni opinion deta hai (like predicting salary or marks),
# aur Random Forest un sabka average nikal ke stable, accurate prediction deta hai.


# ---

# 📊 Example to Understand

# Maan le tu predict kar raha hai kisi employee ka Salary based on experience:

# Tree 1 predict kare: ₹70,000

# Tree 2 predict kare: ₹72,000

# Tree 3 predict kare: ₹75,000
# ➡ Final prediction = Average = ₹72,333


# Iss se Random Forest:

# Outliers ka effect kam karta hai

# Accuracy improve karta hai

# Overfitting reduce karta hai (compared to single Decision Tree)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Dataset
data = {
  "Experience":
  [1,2,3,4,5,6,7,8,9,10],
  "Salary":
  [25000,28000,35000,40000,45000,50000,65000,70000,85000,100000]
}
df=pd.DataFrame(data)

# Features and Target
X = df[["Experience"]]
y = df["Salary"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error :", mean_squared_error(y_test, y_pred))
print("R2 score :", r2_score(y_test,y_pred))

# Feature Importance Visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance in Random Forest Regressor")
plt.show()

# Predicted Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred, color='red', label='Predicted Data')
plt.xlabel("Expericence(Years)")
plt.ylabel("Salary(INR)")
plt.legend()
plt.title("Actual vs Predicted Salary")
plt.show()

# Explanation in Simple Words

# 1. Dataset: We have experience (in years) and salary (in INR).


# 2. Model Creation: RandomForestRegressor() makes 100 decision trees (n_estimators=100).


# 3. Training: Each tree learns different patterns from data.


# 4. Prediction: Each tree predicts salary → model averages all predictions.


# 5. Evaluation:

# Mean Squared Error → measures how far predicted values are from actual salaries (lower = better).

# R² Score → tells how well the model fits the data (closer to 1 = better).



# 6. Visualization:

# Feature importance shows how much "Experience" affects Salary.

# Scatter plot shows actual vs predicted points.





# ---

# 🧠 Real-World Applications

# Salary or house price prediction

# Predicting student marks

# Sales forecasting

# Stock value prediction