import scipy.io
import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD

# 1. Load Data (Treating the whole dataset as one population)
file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    data = scipy.io.loadmat(file_path)
    X = data['X']
    y = data['y'].ravel()

    # 2. Calculate Global Contamination
    # In a pure statistical sense, you'd guess this, but here we use the ground truth
    contam = (y == 1).sum() / len(y)

    # 3. Initialize MCD
    # This is now acting as a robust estimator of the dataset's parameters
    clf = MCD(contamination=contam, random_state=42)

    # 4. "Fit" becomes "Estimate"
    # The algorithm finds the h-subset with the lowest determinant covariance
    clf.fit(X)

    # 5. Extract results for the SAME data points
    # labels_ contains 0 for inliers and 1 for outliers based on the whole set
    predictions = clf.labels_
    anomaly_scores = clf.decision_scores_

    # 6. Metrics
    f1_macro = f1_score(y, predictions, average='macro')
    auc_roc = roc_auc_score(y, anomaly_scores)
    auprc = average_precision_score(y, anomaly_scores)

    print(f"--- Statistical MCD Analysis (Whole Dataset) ---")
    print(f"Total Samples:    {len(X)}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"ROC-AUC Score:    {auc_roc:.4f}")
    print(f"AUPRC Score:      {auprc:.4f}")