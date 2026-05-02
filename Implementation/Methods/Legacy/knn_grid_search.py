import scipy.io
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def start_knn_grid_search(dataset_name='wine', num_splits=500):
    dl = Data_Loader()

    # Define the range of k values to test
    k_values = [5]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(file_path):
        print(f"FEHLER: Datei {file_path} nicht gefunden.")
        return

    # List to store the aggregated results for each k
    grid_summary = []

    print(f"--- Starte k-NN Grid Search: {dataset_name} ---")
    print(f"Testing k in: {k_values}")

    for k in k_values:
        print(f"\nTesting k = {k} ({num_splits} splits)...")
        results = {'f1': [], 'auc': [], 'auprc': []}

        for i in range(num_splits):
            try:
                # 1. Load Data
                train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

                X_train_np = train_data.numpy()
                X_test_np = test_data.numpy()
                y_test_np = test_labels.numpy().ravel()

                # 2. Scaling
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_np)
                X_test_scaled = scaler.transform(X_test_np)

                # 3. Model Training
                clf = KNN(n_neighbors=k, method='largest')
                clf.fit(X_train_scaled)

                # 4. Scores and Top-N Threshold Logic
                test_scores = clf.decision_function(X_test_scaled)
                n_anomalies = int(np.sum(y_test_np))

                # Handle cases with 0 anomalies in a split if necessary
                if n_anomalies == 0:
                    continue

                top_indices = np.argsort(test_scores)[-n_anomalies:]
                test_labels_pred = np.zeros_like(y_test_np)
                test_labels_pred[top_indices] = 1

                # 5. Calculate metrics
                results['f1'].append(f1_score(y_test_np, test_labels_pred))
                results['auc'].append(roc_auc_score(y_test_np, test_scores))
                results['auprc'].append(average_precision_score(y_test_np, test_scores))

            except Exception as e:
                print(f"Fehler in k={k}, Split {i}: {e}")

        # Aggregate metrics for this k
        grid_summary.append({
            'k': k,
            'f1_mean': np.mean(results['f1']),
            'f1_std': np.std(results['f1']),
            'auc_mean': np.mean(results['auc']),
            'auprc_mean': np.mean(results['auprc'])
        })

    # --- Export and Summary ---
    df_grid = pd.DataFrame(grid_summary)
    output_csv = f"knn_grid_search.csv"
    df_grid.to_csv(output_csv, index=False)

    print("\n" + "-" * 65)
    print(f" GRID SEARCH RESULTS: {dataset_name.upper()} ".center(65, " "))
    print("-" * 65)

    # Format the table for readability
    print(df_grid.to_string(index=False, formatters={
        'f1_mean': '{:.4f}'.format,
        'f1_std': '{:.4f}'.format,
        'auc_mean': '{:.4f}'.format,
        'auprc_mean': '{:.4f}'.format
    }))

    print("-" * 65)
    best_f1 = df_grid.loc[df_grid['f1_mean'].idxmax()]
    best_auprc = df_grid.loc[df_grid['auprc_mean'].idxmax()]

    print(f"Best by F1:    k={int(best_f1['k'])} (F1: {best_f1['f1_mean']:.4f})")
    print(f"Best by AUPRC: k={int(best_auprc['k'])} (AUPRC: {best_auprc['auprc_mean']:.4f})")
    print(f"Results saved to: {output_csv}")
    print("=" * 65)


if __name__ == "__main__":
    # Reduced num_splits to 100 for grid search to keep execution time reasonable
    start_knn_grid_search(dataset_name='wine', num_splits=500)