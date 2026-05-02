import os
import scipy.io
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', '../wine.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# 2. Define Hyperparameter Grid
param_grid = {
    'n_neighbors': [5, 10, 15, 20, 27, 35, 50],
    'method': ['largest', 'mean', 'median']
}

outlier_ratio = 0.077
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_auc = -1
best_params = {}

# Header for results table
print(f"{'k':<4} | {'Method':<8} | {'Mean AUC':<10} | {'Mean AUPRC':<10} | {'Mean F1':<10}")
print("-" * 55)

# 3. Tuning Loops
for k in param_grid['n_neighbors']:
    for method in param_grid['method']:
        fold_aucs = []
        fold_auprcs = []
        fold_f1s = []

        for train_index, test_index in kf.split(X):
            # Split and Scale
            X_train, X_test = X[train_index], X[test_index]
            y_test = y[test_index]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize Model
            clf = KNN(n_neighbors=k,
                      method=method,
                      contamination=outlier_ratio,
                      n_jobs=-1)
            clf.fit(X_train_scaled)

            # Get scores for AUC/AUPRC and labels for F1
            test_scores = clf.decision_function(X_test_scaled)
            test_labels = clf.predict(X_test_scaled)

            # 4. Calculate Metrics per Fold
            fold_aucs.append(roc_auc_score(y_test, test_scores))
            fold_auprcs.append(average_precision_score(y_test, test_scores))
            fold_f1s.append(f1_score(y_test, test_labels))

        # 5. Average results across folds
        mean_auc = np.mean(fold_aucs)
        mean_auprc = np.mean(fold_auprcs)
        mean_f1 = np.mean(fold_f1s)

        print(f"{k:<4} | {method:<8} | {mean_auc:.4f}     | {mean_auprc:.4f}      | {mean_f1:.4f}")

        # Update best configuration based on AUC
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = {'n_neighbors': k, 'method': method, 'auprc': mean_auprc, 'f1': mean_f1}

# 6. Final Summary
print("-" * 55)
print(f"BEST CONFIGURATION FOUND:")
print(f"Parameters: {best_params['n_neighbors']} neighbors, {best_params['method']} method")
print(f"ROC-AUC:    {best_auc:.4f}")
print(f"AUPRC:      {best_params['auprc']:.4f}")
print(f"F1 Score:   {best_params['f1']:.4f}")