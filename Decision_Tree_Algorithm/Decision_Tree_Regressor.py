# Decision Tree Regressor ek supervised learning algorithm hai jo continuous values (like Salary, Price, Marks, etc.) predict karta hai.

# Classification → Yes/No type output

# Regression → Number output (float or int)


# Algorithm dataset ko “if-else” rules me todta hai, jaise:

# > If experience < 5 → Salary ≈ 40k
# Else → Salary ≈ 80k



# Yani, tree data ko small decision nodes me divide karta hai aur har node ek prediction deta hai.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Dataset
data = {
  "Experience":
  [1,2,3,4,5,6,7,8,9,10],
  "Salary":
  [25000,28000,35000,40000,45000,50000,65000,70000,85000,100000]
}

df = pd.DataFrame(data)

# Step 2 : Feature & Target
X = df[["Experience"]]  # Independent variable
Y = df["Salary"]  #Dependent variable

#Step 3 : Train-test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Step 4 : Mode Training
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train,Y_train)

# Step 5 : Prediction
Y_pred = model.predict(X_test)

# Step 6 : Evaluation
print("Mean Squared Error :", mean_squared_error(Y_test,Y_pred))
print("R2 Score :", r2_score(Y_test,Y_pred))

# Step 7 : Visualisation
plt.figure(figsize=(10,6))
tree.plot_tree(model,feature_names=["Experience"],filled=True)
plt.show()