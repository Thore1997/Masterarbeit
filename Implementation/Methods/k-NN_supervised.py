import os
import scipy.io
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler # Changed from MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# Method 1: Summing boolean values (True counts as 1)
num_ones = np.sum(y == 1)

# Method 2: Calculating the percentage
outlier_ratio = num_ones / len(y)

print(f"Total instances in y:      {len(y)}")
print(f"Number of '1' instances:   {num_ones}")
print(f"Actual Outlier Ratio:      {outlier_ratio:.4f}")