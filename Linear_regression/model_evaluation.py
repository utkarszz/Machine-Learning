from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
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
print("Actual hours(X_test):", X_test.values)




# Visualization
plt.scatter(X,Y,color="blue", label = "Data points")
plt.plot(X, model.predict(X), color = "red",label = "Best fit line")
plt.xlabel("hours studied")
plt.ylabel("Marks")
plt.legend()
plt.show()

mse = mean_squared_error(Y_test,Y_pred)
r2 = r2_score(Y_test,Y_pred)

print("Mean Squared Error :" ,mse)
print("R2 Score (Accuracy):" ,r2)

# r2 score 0 se 1 k beech hota haii
# 1 = perfect fit 
# 0 = bikul relation nhii

# Haan, aapka samajhna bilkul sahi hai, lekin thoda sa technical clarification zaruri hai.
# Here's the breakdown in Hinglish:
# Aapne Kya Kiya (What You Did)
# Aapki baat bilkul correct hai. Jab aapne data split kiya, toh 20% data testing ke liye alag rakh diya.

# 1.Training stage (model training)
# model.fit(X_train, Y_train)
# Model ne 80% data (8 rows) se X (Hours) aur Y (Marks) ke beech ka relationship seekha: Marks = 10 *Hours

# 2.Testing state (Prediction)

# Aapke paas 20% testing data tha, jismein Xtest(Hours) aur Y test(Actual Marks) dono the.
# Aapne model ko sirf Hours ka actual data bheja: X_test

#  Conclusion: Input for Prediction
# Aapka statement 100\% sahi hai:
# "Mtlb jo actual data 20% bheja gya woh hrs ka gya"
# Aapne prediction ke liye actual hours (i.e., X_test ka data bheja, aur model ne uske basis par Predicted Marks Y_pred calculate kiye.
# Phir, aapne us Predicted Marks ki tulna (comparison) Actual Marks Y_test se ki yeh dekhne ke liye ki model ne kitna achha perform kiya (R=1).


