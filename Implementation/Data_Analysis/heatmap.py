import scipy.io
import scipy.stats as stats  # Added this for Spearman
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')
data = mat['X']

corr_matrix_spearman, _ = stats.spearmanr(data)
mask_s = np.triu(np.ones_like(corr_matrix_spearman, dtype=bool))
feature_name = [f"Feature {i+1}" for i in range(data.shape[1])]

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix_spearman,
    mask=mask_s,
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

plt.tight_layout()
plt.show()