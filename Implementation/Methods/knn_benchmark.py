import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def start_semi_knn_benchmark(dataset_name='wineori', num_splits=500):
    dl = Data_Loader()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(file_path):
        print(f"FEHLER: Datei {file_path} nicht gefunden.")
        return

    results = {'f1': [], 'auc': [], 'auprc': []}
    all_y_true = []
    all_y_scores = []

    print(f"--- Starte Paper-Reproduction k-NN Benchmark: {dataset_name} ---")
    print(f"Konfiguration: {num_splits} Splits, k=5, Scaling=RobustScaler, Threshold=Top-N")

    for i in range(num_splits):
        try:
            # 1. Daten laden
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

            X_train_np = train_data.numpy()
            X_test_np = test_data.numpy()
            y_test_np = test_labels.numpy().ravel()

            # 2. Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_np)
            X_test_scaled = scaler.transform(X_test_np)

            # 3. Model Training
            # Contamination ist hier egal, da wir sie später manuell überschreiben
            clf = KNN(n_neighbors=5, method='largest')
            clf.fit(X_train_scaled)

            # 4. DYNAMISCHER THRESHOLD (Paper-Logik)
            # Wir holen die rohen Distanz-Scores
            test_scores = clf.decision_function(X_test_scaled)

            # Zähle echte Anomalien im Testset (N)
            n_anomalies = int(np.sum(y_test_np))

            # Wähle exakt die N höchsten Scores als Anomalien
            # argsort sortiert aufsteigend -> die letzten N Indizes sind die Top-Scores
            top_indices = np.argsort(test_scores)[-n_anomalies:]

            # Erstelle Vorhersage-Vektor: Alles 0, Top-N sind 1
            test_labels_pred = np.zeros_like(y_test_np)
            test_labels_pred[top_indices] = 1

            # --- Ende Paper-Logik ---

            all_y_true.append(y_test_np)
            all_y_scores.append(test_scores)

            # 5. Metriken speichern
            results['f1'].append(f1_score(y_test_np, test_labels_pred))
            results['auc'].append(roc_auc_score(y_test_np, test_scores))
            results['auprc'].append(average_precision_score(y_test_np, test_scores))

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # --- Speicher- und Export-Logik (unverändert) ---
    if len(all_y_true) > 0:
        if not os.path.exists('Results'): os.makedirs('Results')
        final_y_true = np.concatenate(all_y_true)
        final_y_scores = np.concatenate(all_y_scores)
        np.savez('Results/scores_knn.npz', y_true=final_y_true, y_scores=final_y_scores)

    if len(results['f1']) > 0:
        df_results = pd.DataFrame({
            'split': list(range(1, len(results['f1']) + 1)),
            'f1_score': results['f1'],
            'roc_auc': results['auc'],
            'auprc': results['auprc']
        })
        df_results.to_csv(f"knn_results.csv", index=False, sep=';')
        print(f"\n" + "=" * 45)
        print(f"ERGEBNISSE MIT TOP-N THRESHOLD ({dataset_name})")
        print("-" * 45)
        print(f"F1-Score: {df_results['f1_score'].mean():.4f} ± {df_results['f1_score'].std():.4f}")
        print(f"ROC-AUC:  {df_results['roc_auc'].mean():.4f} ± {df_results['roc_auc'].std():.4f}")
        print(f"AUPRC:    {df_results['auprc'].mean():.4f} ± {df_results['auprc'].std():.4f}")
        print("=" * 45)


if __name__ == "__main__":
    start_semi_knn_benchmark(dataset_name='wine', num_splits=500)