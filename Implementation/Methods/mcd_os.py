import scipy.io
import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD

file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')
data = scipy.io.loadmat(file_path)
X = data['X']
y = data['y'].ravel()



clf = MCD(contamination=0.0775, random_state=42)
clf.fit(X)


predictions = clf.labels_
anomaly_scores = clf.decision_scores_


f1 = f1_score(y, predictions)
f1_macro = f1_score(y, predictions, average='macro')
auc_roc = roc_auc_score(y, anomaly_scores)
auprc = average_precision_score(y, anomaly_scores)

print(f"--- Results MCD ---")
print(f"Total Samples:      {len(X)}")
print(f"F1 (Macro) Score:   {f1_macro:.4f}")
print(f"F1 (binary) Score:  {f1:.4f}")
print(f"ROC-AUC Score:      {auc_roc:.4f}")
print(f"AUPRC Score:        {auprc:.4f}")