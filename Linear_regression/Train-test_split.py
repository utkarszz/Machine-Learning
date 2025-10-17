# ML mei dataset ko hamesha 2 parts mei todte haii
# 1st Training set = model ko sikhane k liye
# 2nd Testing set = model ko test krne k liye
import pandas as pd
from sklearn.model_selection import train_test_split

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

print("Train set size :",X_train.shape)
print("Test set size :",X_test.shape)