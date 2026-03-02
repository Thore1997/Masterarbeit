import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')
data = mat['X']

corr_matrix = np.corrcoef(data, rowvar=False)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

feature_name = [f"Feature {i+1}" for i in range(data.shape[1])]

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".6f",
    cmap='coolwarm',
    center=0,
    annot_kws={"size": 9},
    square=True,
    linewidths=.5,           # Added comma here
    xticklabels=feature_name, # Moved inside the parenthesis
    yticklabels=feature_name  # Moved inside the parenthesis
)                            # Moved the closing parenthesis to HERE

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

######## Spearman Correlation Heatmap #######
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

feature_name = [f"Feature {i+1}" for i in range(data.shape[1])]

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".6f",
    cmap='coolwarm',
    center=0,
    annot_kws={"size": 9},
    square=True,
    linewidths=.5,
    xticklabels=feature_name,
    yticklabels=feature_name
)

plt.title("Spearman Rank Correlation Heatmap")
plt.tight_layout()
plt.show()