#  What is LDA? (Linear Discriminant Analysis)

# Ye ek Supervised Dimensionality Reduction technique hai.
# Supervised = Labels use karta hai
# Dimensionality Reduction = Features kam karta hai

#  Objective:

# > Classes ko maximum separate karna with least features.




# ---

#  Real-Life Example

# Imagine tumhare paas students ke marks wale 4 subjects ka data hai
# (Physics, Chem, Math, English)

# Aur tum sirf yeh predict karna chahte ho:

# Pass (1)

# Fail (0)


#  Par 4 subjects use karne ki jagah
# LDA data ko 2 features me convert kar dega
# but Pass vs Fail ka difference zyada highlight hoga

# Jaise ek straight boundary
# Pass students ek side
# Fail students ek side


# ---





# ---

#  LDA Kya Karta Hai?

# Mathematical idea:

#  Each class ka mean vector nikalta hai
#  Puri dataset ka mean nikalta hai
#  Two matrices banata hai:

# Within-Class Scatter → Class ke andar spread

# Between-Class Scatter → Class ke beech difference


#  Ratio maximize karta hai:

# > Between-Class difference ↑
# Within-Class scatter ↓



# Isliye graph me clusters Door-Door & clear dikhte hain 


# ---

#  LDA Output Kaha Use Hota Hai?

# Use Case	Why LDA works well

# Face Recognition	Faces belong to distinct classes
# Cancer Classification (Benign/Malignant)	Medical labels clearly separate
# Fraud Detection	Fraud vs Genuine data separation
# Customer Segmentation	Purchase behavior ke basis par groups



# ---

#  LDA Benefits

# Benefit	Meaning

# Speed Boost	Features kam → Fast model
# Overfitting reduce	Noise remove
# Better classification	Separation improved
# Visualization	2D ya 3D me plot possible



# ---

#  Rule of Thumb

# > Jab labels ho → LDA better
# Jab sirf data ho → PCA better




import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Loading Dataset
iris= datasets.load_iris()
X= iris.data
y=iris.target

# Standarizing data
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)

# Applying LDA- 2components(for visualization)
lda = LDA(n_components=2)
X_lda= lda.fit_transform(X_scaled,y)

# Plotting
plt.figure(figsize=(8,6))
for label, color in zip(np.unique(y),['blue','green','red']): plt.scatter(X_lda[y== label,0], X_lda[y== label, 1], label = f"Class{label}", color=color)

plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.title("LDA Visualization (Class Separation)")
plt.legend()
plt.show()