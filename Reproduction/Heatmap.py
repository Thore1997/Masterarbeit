import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
mat = scipy.io.loadmat('Data/thyroid_processed_dataset.mat')
data = mat['X']

# 2. Define your labels
labels = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

# 3. Calculate Correlation Matrix
# rowvar=False because your columns are the variables
corr_matrix = np.corrcoef(data, rowvar=False)

# 4. Plot
plt.figure(figsize=(12, 10))

# Using '.3e' for scientific notation (e.g., 1.234e-05)
# This prevents small correlations from being rounded to zero
sns.heatmap(corr_matrix,
            annot=True,
            fmt=".6f",
            cmap='coolwarm',
            center=0,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 9})

plt.title("Correlation Heatmap")
plt.tight_layout()

# 5. Display
plt.show()