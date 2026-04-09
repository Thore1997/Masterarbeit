import os
import scipy.io
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 1. Load Data
file_path = os.path.join('Reproduction', 'Data', 'wine.mat')
mat_data = scipy.io.loadmat(file_path)
X = mat_data['X']
y = mat_data['y'].ravel()

# 2. Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize and Fit Model
outlier_ratio = 0.077
clf = KNN(
    n_neighbors=20,
    method='largest',
    metric='euclidean',
    contamination=outlier_ratio,
    n_jobs=-1
)
clf.fit(X_scaled)

# 4. Extract Results
scores = clf.decision_scores_
labels = clf.labels_

# --- NEW: Extract Outliers ---
# Get the indices where the label is 1
outlier_indices = np.where(labels == 1)[0]

# Extract the actual data points defined as outliers
outliers_data = X[outlier_indices]
outlier_scores = scores[outlier_indices]

# 5. Evaluate
auc = roc_auc_score(y, scores)
auprc = average_precision_score(y, scores)
f1 = f1_score(y, labels)

# 6. Print Results
print("--- Unsupervised k-NN Results ---")
print(f"ROC-AUC: {auc:.4f}")
print(f"AUPRC:   {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\n--- Outlier Detection Summary ---")
print(f"Total points classified as outliers: {len(outlier_indices)}")
print(f"Indices of outliers: {outlier_indices}")

################################################
import os
import torch
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# Importiere deinen existierenden Data_Loader
from data_loader import Data_Loader


def start_unsup_knn_benchmark(dataset_name='wine', num_splits=100):
    dl = Data_Loader()

    # Pfad-Logik entsprechend deiner Repository-Struktur [cite: 1028]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(mat_file_path):
        print(f"FEHLER: Datei {mat_file_path} nicht gefunden.")
        return

    # Listen für Metriken (für Mittelwert und SD in Tabelle 1) [cite: 1122, 1127]
    all_f1, all_auc, all_prc = [], [], []

    # Konstante für den k-NN (wie in deiner Thesis beschrieben) [cite: 529]
    outlier_ratio = 0.077

    print(f"--- Starte Unsupervised k-NN Benchmark ({num_splits} Splits) ---")

    for i in range(num_splits):
        try:
            # 1. Daten laden (identischer 50/50 Split) [cite: 1059, 1086]
            # Wir ignorieren train_data bewusst, da dieser Ansatz unsupervised ist
            _, test_data, test_labels = dl.build_train_test_generic_matfile(mat_file_path)

            # Umwandlung für NumPy/PyOD
            X_test = test_data.numpy()
            y_test = test_labels.numpy().ravel()

            # 2. Scaling (RobustScaler basierend auf den Test-Daten) [cite: 515, 1103]
            scaler = RobustScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            # 3. Modell-Initialisierung (Unsupervised) [cite: 499]
            # Das Modell "sieht" nur die Test-Daten
            clf = KNN(
                n_neighbors=5,
                method='largest',
                metric='euclidean',
                contamination=outlier_ratio,
                n_jobs=-1
            )

            # Fit & Predict auf derselben Datenbasis (X_test) [cite: 499, 503]
            clf.fit(X_test_scaled)

            scores = clf.decision_scores_
            labels = clf.labels_

            # 4. Metriken speichern
            all_f1.append(f1_score(y_test, labels))
            all_auc.append(roc_auc_score(y_test, scores))
            all_prc.append(average_precision_score(y_test, scores))

            if (i + 1) % 20 == 0:
                print(f"Split {i + 1}/{num_splits} verarbeitet...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # --- FINALE AUSGABE (für Tabelle 1 in deinem PDF) [cite: 1126, 1127] ---
    if len(all_f1) > 0:
        print(f"\n" + "=" * 45)
        print(f"UNSUPERVISED k-NN ERGEBNISSE ({dataset_name})")
        print("-" * 45)
        print(f"F1-Score: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        print(f"ROC-AUC:  {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
        print(f"AUPRC:    {np.mean(all_prc):.4f} ± {np.std(all_prc):.4f}")
        print("=" * 45)


if __name__ == "__main__":
    # Standardmäßig 100 oder 500 Durchläufe [cite: 1065]
    start_unsup_knn_benchmark('wine', num_splits=100)