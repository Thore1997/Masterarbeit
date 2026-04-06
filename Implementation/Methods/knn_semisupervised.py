import scipy.io
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from pyod.models.knn import KNN

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', 'wine.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# Separate Normals and Anomalies
X_normals = X[y == 0]
y_normals = y[y == 0]
X_anomalies = X[y == 1]
y_anomalies = y[y == 1]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Added 'f1_macro' to the metrics dictionary
fold_metrics = {'f1': [], 'f1_macro': [], 'roc_auc': [], 'auprc': []}

print(f"--- PyOD k-NN 5-Fold Cross-Validation ---")

# 3. CV Loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X_normals)):
    X_train_fold = X_normals[train_idx]
    X_test_norm_fold = X_normals[test_idx]

    X_test_fold = np.vstack([X_test_norm_fold, X_anomalies])
    y_test_fold = np.hstack([np.zeros(len(X_test_norm_fold)), np.ones(len(X_anomalies))])

    # 4. Scaling
    scaler = RobustScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_test_fold = scaler.transform(X_test_fold)

    # 5. Model Training
    clf = KNN(n_neighbors=5, method='largest', contamination=0.077)
    clf.fit(X_train_fold)

    # 6. Predictions & Scores
    predictions = clf.predict(X_test_fold)
    test_scores = clf.decision_function(X_test_fold)

    # Record Metrics
    fold_metrics['f1'].append(f1_score(y_test_fold, predictions))
    fold_metrics['f1_macro'].append(f1_score(y_test_fold, predictions, average='macro'))
    fold_metrics['roc_auc'].append(roc_auc_score(y_test_fold, test_scores))
    fold_metrics['auprc'].append(average_precision_score(y_test_fold, test_scores))

    print(f"Fold {fold + 1}: ROC-AUC = {fold_metrics['roc_auc'][-1]:.4f}")
    print(f"Fold {fold + 1}: F1 (Binary) = {fold_metrics['f1'][-1]:.4f}")
    print(f"Fold {fold + 1}: F1 (Macro) = {fold_metrics['f1_macro'][-1]:.4f}")

# 7. Final Results
print(f"\n--- Final Average Results ---")
for metric, scores in fold_metrics.items():
    print(f"Mean {metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")