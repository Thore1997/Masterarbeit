import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def run_extended_grid_search(dataset_name='wine', num_splits=500):
    dl = Data_Loader()

    # Removed 'robust' from the parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20],
        'scaler': ['standard', 'minmax']
    }

    # Removed RobustScaler from the factory
    scaler_factory = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(file_path):
        print(f"FEHLER: Datei {file_path} nicht gefunden.")
        return

    grid_results = []

    for params in combinations:
        k = params['n_neighbors']
        s_name = params['scaler']

        print(f"Testing: Scaler={s_name:8} | k={k:2}")
        results = {'f1': []}

        for i in range(num_splits):
            try:
                # 1. Fresh Data for every split
                train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

                # 2. Dynamic Scaler selection
                scaler = scaler_factory[s_name]()

                # 3. Fit on Train, Transform Test
                X_train = scaler.fit_transform(train_data.numpy())
                X_test = scaler.transform(test_data.numpy())
                y_test = test_labels.numpy().ravel()

                # Model
                clf = KNN(n_neighbors=k)
                clf.fit(X_train)

                test_scores = clf.decision_function(X_test)
                n_anomalies = int(np.sum(y_test))

                # Calculate y_pred based on top anomaly scores
                top_indices = np.argsort(test_scores)[-n_anomalies:]
                y_pred = np.zeros_like(y_test)
                y_pred[top_indices] = 1

                results['f1'].append(f1_score(y_test, y_pred))

            except Exception as e:
                print(f"Error during iteration: {e}")
                continue

        # Store mean and std for the combination
        grid_results.append({
            'scaler': s_name,
            'k': k,
            'f1_mean': np.mean(results['f1']) if results['f1'] else 0,
            'f1_std': np.std(results['f1']) if results['f1'] else 0
        })

    # --- Data Processing: Get Top 5 Per Scaler ---
    df_grid = pd.DataFrame(grid_results)

    # Group by scaler and take the top 5 largest f1_mean values for each group
    df_top_5 = df_grid.sort_values(['scaler', 'f1_mean'], ascending=[True, False])
    df_top_5 = df_top_5.groupby('scaler').head(5).reset_index(drop=True)

    # --- CSV Export ---
    csv_name = f"grid_search_{dataset_name}_top_results.csv"
    df_top_5.to_csv(csv_name, index=False)

    # --- Reporting ---
    print("\n" + "-" * 60)
    print(f" TOP 5 RESULTS PER NORMALIZATION: {dataset_name.upper()} ".center(60, " "))
    print("-" * 60)

    # Display grouped results in console
    for scaler_type in df_top_5['scaler'].unique():
        print(f"\n>>> Scaler: {scaler_type.upper()}")
        subset = df_top_5[df_top_5['scaler'] == scaler_type]
        print(subset[['k', 'f1_mean', 'f1_std']].to_string(index=False, formatters={
            'f1_mean': '{:.4f}'.format,
            'f1_std': '{:.4f}'.format
        }))

    print("\n" + "-" * 60)
    print(f"SUCCESS: Results saved to {csv_name}")
    print("=" * 60)


if __name__ == "__main__":
    run_extended_grid_search(dataset_name='wine', num_splits=500)