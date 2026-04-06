import os
import scipy.io
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', 'wine.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# 2. Setup Cross-Validation
outlier_ratio = 0.077
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics for each fold
auc_list = []
auprc_list = []
f1_list = []
f1_macro_list = []

# 3. CV Loop
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_test = y[test_index]

    # 4. Scaling (Inductive approach: fit on train, transform test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Initialize and Fit Model
    clf = KNN(
        n_neighbors=20,  # Increased from 5
        method='mean',  # Changed from 'largest'
        metric='euclidean',
        contamination=outlier_ratio,
        n_jobs=-1)

    # Fit on training fold
    clf.fit(X_train_scaled)

    # 6. Predict on Test Fold
    test_scores = clf.decision_function(X_test_scaled)
    test_labels = clf.predict(X_test_scaled)

    # 7. Evaluate Fold
    fold_auc = roc_auc_score(y_test, test_scores)
    fold_auprc = average_precision_score(y_test, test_scores)
    fold_f1 = f1_score(y_test, test_labels)
    fold_f1_macro = f1_score(y_test, test_labels, average='macro')

    auc_list.append(fold_auc)
    auprc_list.append(fold_auprc)
    f1_list.append(fold_f1)
    f1_macro_list.append(fold_f1_macro)

    print(f"Fold {fold} | AUC: {fold_auc:.4f} | AUPRC: {fold_auprc:.4f} | F1: {fold_f1:.4f} | Macro-F1: {fold_f1_macro:.4f}")

# 8. Final Results (Mean and Std Dev)
print("-" * 45)
print(f"FINAL MEAN ROC-AUC:  {np.mean(auc_list):.4f} (+/- {np.std(auc_list):.4f})")
print(f"FINAL MEAN AUPRC:    {np.mean(auprc_list):.4f} (+/- {np.std(auprc_list):.4f})")
print(f"FINAL MEAN F1:       {np.mean(f1_list):.4f} (+/- {np.std(f1_list):.4f})")
print(f"FINAL MEAN MACRO-F1: {np.mean(f1_macro_list):.4f} (+/- {np.std(f1_macro_list):.4f})")