#  What is Kernel in SVM? (Simple explanation)

# SVM ka kaam hota hai data classes ke beech best boundary (decision boundary) banane ka.

# But problem:
# Har data linearly separable nahi hota.

#  Isliye SVM kernel use karta hai — jisse data ko higher dimensions me map karke separable banaya ja sake.


# ---

#  Common Kernels

# Kernel	Where used	Simple Meaning

# linear	Jab data straight line se separate ho jaye	Straight boundary
# poly (polynomial)	Curved boundary	Data me curves ho
# rbf (Gaussian)	Complex shapes	Random scattered data



# ---

# Visualization image samajh lo:

# Linear Kernel → Straight line boundary
# Poly Kernel   → Curve boundary
# RBF Kernel    → Irregular shape boundary


# # Bilkul sahi — tumne kernel use kar liya,(SVM.py mei kernal use kiya bass tuning nhi)
# par “kernel tuning” ≠ just kernel change.

# Tumne abhi sirf ye kiya hai:

# model = SVC(kernel="rbf")

#  Kernel used
#  Kernel tuned nahi kiya

# Kernel tuning ka matlab hota hai parameters ke saath khelna:

# kernel  (linear, rbf, poly)

# C

# gamma


# Most important tuning C and gamma hai.
# Ye dono decide karte hain ki boundary kaise banegi.


# ---

#  C kya hota hai?

#  C = Model kitni galti allow karega

# C value	Result

# High C	Model errors allow nahi karta → boundary tight → overfitting risk
# Low C	Model thoda error allow karta → boundary smooth → generalizes better


# Simple line:

# > C bada → rigid model
# C chhota → flexible model




# ---

#  Gamma kya hota hai?

#  Gamma = ek point ka influence kitni door tak rahega

# Gamma value	Result

# High γ (gamma)	Boundary zig-zag, complex curve → overfitting
# Low γ (gamma)	Boundary smooth → simpler model


# Simple line:

# > Gamma bada → sharp curves
# Gamma chhota → smooth curves

# Bilkul sahi — tumne kernel use kar liya,
# par “kernel tuning” ≠ just kernel change.

# Tumne abhi sirf ye kiya hai:

# model = SVC(kernel="rbf")

#  Kernel used
#  Kernel tuned nahi kiya

# Kernel tuning ka matlab hota hai parameters ke saath khelna:

# kernel  (linear, rbf, poly)

# C

# gamma


# Most important tuning C and gamma hai.
# Ye dono decide karte hain ki boundary kaise banegi.


# ---

#  C kya hota hai?

#  C = Model kitni galti allow karega

# C value	Result

# High C	Model errors allow nahi karta → boundary tight → overfitting risk
# Low C	Model thoda error allow karta → boundary smooth → generalizes better


# Simple line:

# > C bada → rigid model
# C chhota → flexible model




# ---

#  Gamma kya hota hai?

#  Gamma = ek point ka influence kitni door tak rahega

# Gamma value	Result

# High γ (gamma)	Boundary zig-zag, complex curve → overfitting
# Low γ (gamma)	Boundary smooth → simpler model


# Simple line:

# > Gamma bada → sharp curves
# Gamma chhota → smooth curves

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Loading iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using first two features for easy visualization
Y = iris.target

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Function to draw decision boundary
def plot_svm(kernel, C=1, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, Y_train)

    # Creating a meshgrid ie grid for plotting decision boundary
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.02), np.arange(Y_min, Y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="coolwarm", edgecolors="black")
    plt.title(f"kernel={kernel}, C={C}, gamma={gamma}")


# Visualization

plt.figure(figsize=(15, 10))

# kernal (kernel) comparison
plt.subplot(3, 3, 1)
plot_svm(kernel="linear")

plt.subplot(3, 3, 2)
plot_svm(kernel="poly")

plt.subplot(3, 3, 3)
plot_svm(kernel="rbf")

# C(strictness) comparison 
plt.subplot(3, 3, 4)
plot_svm(kernel="rbf", C=0.1)  # smooth --> mistakes allow karta h

plt.subplot(3, 3, 5)
plot_svm(kernel="rbf", C=1)    # balanced

plt.subplot(3, 3, 6)
plot_svm(kernel="rbf", C=100)  # rigid --> mistakes allow nhi karta h

# Gamma(curve shape) comparison
plt.subplot(3, 3, 7)
plot_svm(kernel="rbf", gamma=0.01)  # smooth curves

plt.subplot(3, 3, 8)
plot_svm(kernel="rbf", gamma=0.1)   # balanced

plt.subplot(3, 3, 9)
plot_svm(kernel="rbf", gamma=1)     # sharp curves

plt.tight_layout()
plt.show()

# What you learned from this?

# Kernel chooses boundary ka shape (line / curve)

# C controls strictness

# Gamma controls curve sharpness