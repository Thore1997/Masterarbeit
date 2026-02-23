import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Scale
mat = scipy.io.loadmat('Data/thyroid_processed_dataset.mat')
X = mat['X']
# Fix: Ensure we catch the correct label key
y = mat['y'].ravel() if 'y' in mat else mat['Y'].ravel()

feature_names = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
df = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=feature_names)
df['Status'] = y

# 2. Setup the Subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Map status to readable labels for the legend
df['Label'] = df['Status'].map({0: 'Normal', 1: 'Outlier'})
colors = {'Normal': '#3b4cc0', 'Outlier': '#b40426'}

# 3. Generate Density Plots
for i, col in enumerate(feature_names):
    sns.kdeplot(
        data=df,
        x=col,
        hue='Label',
        fill=True,
        palette=colors,
        ax=axes[i],
        alpha=0.4,
        # THIS IS THE KEY: Normalizes each group individually so heights are comparable
        common_norm=False
    )

    axes[i].set_title(f'Density: {col}', fontweight='bold')
    axes[i].set_xlabel('Normalized Value')
    axes[i].set_ylabel('Density')

plt.suptitle('Thyroid Dataset: Normalized Density Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()