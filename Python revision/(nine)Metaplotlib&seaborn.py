# metaplotlib & seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Marks': [85, 92, 78, 90, 88],
}
df = pd.DataFrame(data)
nums = [1, 2, 3, 4, 5]
plt.plot(nums)
plt.plot(nums, [n**2 for n in nums])  # Squared values
plt.show()
sns.histplot(df["Marks"], kde=True)
plt.show()