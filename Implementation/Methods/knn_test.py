import scipy.io
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def start_semi_knn_benchmark(dataset_name='wine', num_splits=500, k=5, method='largest', scaler_type='robust'):
    # 1. Initialize your specific Data_Loader
    dl = Data_Loader()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(file_path):
        print(f"FEHLER: Datei {file_path} nicht gefunden.")
        return

    # Map the console string to the actual Scikit-Learn object
    scaler_map = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    selected_scaler = scaler_map.get(scaler_type.lower(), RobustScaler())

    results = {'f1': [], 'auc': [], 'auprc': []}

    print(f"--- Starte Benchmark: {dataset_name} ---")
    print(f"Params: splits={num_splits}, k={k}, method={method}, scaler={scaler_type}")

    for i in range(num_splits):
        try:
            # 2. Use YOUR Data_Loader's exact split logic
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

            X_train_np = train_data.numpy()
            X_test_np = test_data.numpy()
            y_test_np = test_labels.numpy().ravel()

            # 3. Scaling based on choice
            X_train_scaled = selected_scaler.fit_transform(X_train_np)
            X_test_scaled = selected_scaler.transform(X_test_np)

            # 4. Model Training with chosen k and method
            clf = KNN(n_neighbors=k, method=method)
            clf.fit(X_train_scaled)

            # 5. Paper Logic: Top-N Thresholding
            test_scores = clf.decision_function(X_test_scaled)
            n_anomalies = int(np.sum(y_test_np))
            top_indices = np.argsort(test_scores)[-n_anomalies:]

            test_labels_pred = np.zeros_like(y_test_np)
            test_labels_pred[top_indices] = 1

            # Store metrics
            results['f1'].append(f1_score(y_test_np, test_labels_pred))
            results['auc'].append(roc_auc_score(y_test_np, test_scores))
            results['auprc'].append(average_precision_score(y_test_np, test_scores))

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # Output Final Results
    if results['f1']:
        df = pd.DataFrame(results)
        print(f"\n" + "=" * 45)
        print(f"FINAL STATS ({dataset_name})")
        print("-" * 45)
        print(f"F1-Score: {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
        print(f"ROC-AUC:  {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='k-NN Paper Reproduction')

    # Console Arguments
    parser.add_argument('--dataset', type=str, default='wine')
    parser.add_argument('--splits', type=int, default=500)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--method', type=str, default='largest', choices=['largest', 'mean', 'median'])
    parser.add_argument('--scaler', type=str, default='robust', choices=['robust', 'standard', 'minmax'])

    args = parser.parse_args()

    start_semi_knn_benchmark(
        dataset_name=args.dataset,
        num_splits=args.splits,
        k=args.k,
        method=args.method,
        scaler_type=args.scaler
    )