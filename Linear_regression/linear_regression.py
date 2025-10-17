from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# sample dataset
data={
  "hour_studied":
  [1,2,3,4,5,6,7,8,9,10],
  "Marks":
  [10,20,30,40,50,60,70,80,90,100]
}
df = pd.DataFrame(data)

X = df[["hour_studied"]] #features
Y = df[["Marks"]] #target

# 80% train, 20% test
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# model
model = LinearRegression()
model.fit(X_train, Y_train)

# prediction
Y_pred = model.predict(X_test)
print("Predicted Marks:", Y_pred)
print("Actual Marks:", Y_test.values)

# Visualization
plt.scatter(X,Y,color="blue", label = "Data points")
plt.plot(X, model.predict(X), color = "red",label = "Best fit line")
plt.xlabel("hours studied")
plt.ylabel("Marks")
plt.legend()
plt.show()
