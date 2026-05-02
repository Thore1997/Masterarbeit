import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
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

    results = {'f1': []}
    all_y_true = []
    all_y_scores = []

    print(f"--- Starte Paper-Reproduction k-NN Benchmark: {dataset_name} ---")
    print(f"Konfiguration: {num_splits} Splits, k=5, Scaling=RobustScaler, Threshold=Top-N")

    for i in range(num_splits):
        try:

            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

            X_train_np = train_data.numpy()
            X_test_np = test_data.numpy()
            y_test_np = test_labels.numpy().ravel()

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_np)
            X_test_scaled = scaler.transform(X_test_np)

            clf = KNN(n_neighbors=5, method='largest')
            clf.fit(X_train_scaled)

            test_scores = clf.decision_function(X_test_scaled)

            n_anomalies = int(np.sum(y_test_np))

            top_indices = np.argsort(test_scores)[-n_anomalies:]

            test_labels_pred = np.zeros_like(y_test_np)
            test_labels_pred[top_indices] = 1


            all_y_true.append(y_test_np)
            all_y_scores.append(test_scores)

            results['f1'].append(f1_score(y_test_np, test_labels_pred))

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    if len(results['f1']) > 0:
        df_results = pd.DataFrame({
            'split': list(range(1, len(results['f1']) + 1)),
            'f1_score': results['f1'],

        })
        df_results.to_csv(f"knn_results_minmax.csv", index=False, sep=';')
        print(f"\n" + "=" * 45)
        print(f"ERGEBNISSE MIT TOP-N THRESHOLD ({dataset_name})")
        print("-" * 45)
        print(f"F1-Score: {df_results['f1_score'].mean():.4f} ± {df_results['f1_score'].std():.4f}")
        print("=" * 45)


if __name__ == "__main__":
    start_semi_knn_benchmark(dataset_name='wine', num_splits=500)