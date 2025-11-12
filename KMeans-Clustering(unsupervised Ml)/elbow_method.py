# Elbow method batata hai ki KMeans me best number of clusters (K) kitne hone chahiye.



# K-Means me n_clusters = K hum khud set karte hain.

# But how to decide K = 2, 3, 4, 5 ?

# Elbow method WCSS (Within Cluster Sum of Squares) measure karta hai.


# ---

#  WCSS kya hota hai?

# WCSS = har cluster me points aur centroid ke beech distance ka sum

# Formula samajhne ki zarurat nahi, intuition samajh lo:

# Agar K chhota hoga → group bade aur inaccurate honge → WCSS high

# Agar K badhate jaoge → WCSS kam hota jata hai (clusters tight hote jaate)


# Ek point aata hai jaha improvement slow ho jata hai.

# Wo point = Elbow Point = Best K


# ---

#  Steps of Elbow Method

# 1. K = 1 se K = 10 tak KMeans run karo


# 2. Har K ke liye WCSS store karo


# 3. Graph plot karo → elbow jahan banta hai, wahi best K

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset (mall customers dataset)
data = {
    "Age": [18,22,25,30,35,40,45,50,60,65],
    "Spending_Score":[80,75,60,65,30,25,20,15,40,10]
}
df = pd.DataFrame(data)

X = df[["Age", "Spending_Score"]]
wcss = []
# Trying K from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ gives WCSS

# Plotting the Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS(Inertia)')
plt.title('Elbow Method For Finding Optimal K')
plt.show()