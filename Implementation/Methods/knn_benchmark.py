import scipy.io
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def start_semi_knn_benchmark(dataset_name='wine', num_splits=500):
    """
    Führt einen Semi-Supervised k-NN Benchmark durch.
    Trainiert auf sauberen Normaldaten, testet auf Mix aus Normalen & Anomalien.
    """
    dl = Data_Loader()

    # Pfad-Logik (wie in deinem MCD Skript)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(file_path):
        print(f"FEHLER: Datei {file_path} nicht gefunden.")
        return

    # Speicher für die Metriken
    results = {'f1': [], 'auc': [], 'auprc': []}

    print(f"--- Starte Semi-Supervised k-NN Benchmark: {dataset_name} ---")
    print(f"Konfiguration: {num_splits} Splits, k=5, Scaling=RobustScaler")

    for i in range(num_splits):
        try:
            # 1. Daten laden (50/50 Split)
            # train: nur Normale | test: restliche Normale + alle Anomalien
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

            # Konvertierung für PyOD
            X_train_np = train_data.numpy()
            X_test_np = test_data.numpy()
            y_test_np = test_labels.numpy().ravel()

            # 2. Scaling (Fit nur auf Train!)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_np)
            X_test_scaled = scaler.transform(X_test_np)

            # 3. Model Training (Semi-Supervised)
            # Wir nehmen contamination=0.01, da das Training-Set nominell sauber ist
            clf = KNN(n_neighbors=5, method='largest', contamination=0.154)
            clf.fit(X_train_scaled)

            # 4. Scoring & Predictions
            test_scores = clf.decision_function(X_test_scaled)
            test_labels_pred = clf.predict(X_test_scaled)

            # 5. Metriken speichern
            results['f1'].append(f1_score(y_test_np, test_labels_pred))
            results['auc'].append(roc_auc_score(y_test_np, test_scores))
            results['auprc'].append(average_precision_score(y_test_np, test_scores))

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # --- Finale Auswertung ---
    if len(results['f1']) > 0:
        print(f"\n" + "=" * 45)
        print(f"FINALE SEMI-SUPERVISED k-NN ERGEBNISSE ({dataset_name})")
        print("-" * 45)
        print(f"F1-Score: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
        print(f"ROC-AUC:  {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
        print(f"AUPRC:    {np.mean(results['auprc']):.4f} ± {np.std(results['auprc']):.4f}")
        print("=" * 45)
    else:
        print("Keine erfolgreichen Durchläufe zu protokollieren.")


if __name__ == "__main__":
    # Aufruf der Funktion
    start_semi_knn_benchmark(dataset_name='wine', num_splits=500)