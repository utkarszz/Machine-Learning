import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:",arr.mean(),"Sum:",arr.sum())

# Create a 2D array(Matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:",matrix.shape)
print("Transpose:\n",matrix.T)

# create special arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 2))
rand = np.random.rand(2, 3) # random values between 0 and 1
print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Random:\n", rand)

# notes
# np.array() = basic array creation
# .shape,.T = dimensions and transpose
# np.zeros,np.ones,np.random.rand = array generators