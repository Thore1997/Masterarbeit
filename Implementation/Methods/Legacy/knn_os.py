import os
import scipy.io
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', '../wine.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# 2. Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize and Fit Model
outlier_ratio = 0.077
clf = KNN(
    n_neighbors=25,
    method='largest',
    metric='euclidean',
    contamination=outlier_ratio,
    n_jobs=-1
)
clf.fit(X_scaled)

# 4. Extract Results
scores = clf.decision_scores_
labels = clf.labels_


# 5. Evaluate
auc = roc_auc_score(y, scores)
auprc = average_precision_score(y, scores)
f1 = f1_score(y, labels)

# 6. Print Results
print("--- Unsupervised k-NN Results ---")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"AUPRC:   {auprc:.4f}")



