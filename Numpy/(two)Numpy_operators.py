import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Addition:", a+b)
print("Multiplication:", a*b)
print("Dot Product:",np.dot(a,b))
print("Element-wise Square:",a**2)

# Boolean indexing
nums = np.array([10,20,30,40])
print(nums[nums>20]) # Filter

