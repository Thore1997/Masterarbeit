import scipy.io
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Daten laden
mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')
X = mat['X']
y = mat['y'].ravel()

outlier_fraction = np.sum(y) / len(y)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_values = [1, 3, 5, 10, 20]
all_results = []

for k in k_values:
    metrics = {'auc': [], 'auprc': [], 'f1': []}

    for train_index, test_index in kf.split(X, y):
        X_train_full, X_test = X[train_index], X[test_index]
        y_train_full, y_test = y[train_index], y[test_index]

        # Semi-supervised: Nur Normaldaten ins Training
        X_train_normal = X_train_full[y_train_full == 0]

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_train_normal)

        # Distanzen berechnen
        distances, _ = knn.kneighbors(X_test)
        scores = distances.mean(axis=1)  # Durchschnittsdistanz als Anomalie-Score

        # --- Metriken berechnen ---
        # 1. ROC-AUC
        metrics['auc'].append(roc_auc_score(y_test, scores))

        # 2. AUPRC (Average Precision)
        metrics['auprc'].append(average_precision_score(y_test, scores))

        # 3. F1-Score (Schwellenwert-basiert)
        # Wir nehmen an, dass die obersten 'outlier_fraction' Prozent Outlier sind
        threshold = np.percentile(scores, 100 * (1 - outlier_fraction))
        y_pred = (scores >= threshold).astype(int)
        metrics['f1'].append(f1_score(y_test, y_pred))

    # Ergebnisse mitteln
    res_avg = {m: np.mean(val) for m, val in metrics.items()}
    all_results.append((k, res_avg))
    print(f"k={k:2d} | AUC: {res_avg['auc']:.3f} | AUPRC: {res_avg['auprc']:.3f} | F1: {res_avg['f1']:.3f}")

# Bestes k basierend auf AUPRC finden (oft am stabilsten für Outlier)
best_k = max(all_results, key=lambda x: x[1]['auprc'])[0]
print(f"\nEmpfohlenes k basierend auf AUPRC: {best_k}")