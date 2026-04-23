import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from pyod.models.knn import KNN
from data_loader import Data_Loader


def run_extended_grid_search(dataset_name='wine', num_splits=50):
    dl = Data_Loader()

    param_grid = {
        'n_neighbors': [5, 7, 9, 10, 11, 13, 15, 17, 19, 20],
        'scaler': ['robust', 'standard', 'minmax']
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

        print(f"Testing: Scaler={s_name}, k={k}")
        results = {'f1': []}

        for i in range(num_splits):
            try:
                # 1. Fresh Data for every split
                train_data, test_data, test_labels = dl.build_train_test_generic_matfile(file_path)

                # 2. FIX: Create a FRESH scaler instance here to prevent any state leakage
                if s_name == 'robust':
                    scaler = RobustScaler()
                elif s_name == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()

                # 3. Fit on Train, Transform Test (Proper logic)
                X_train = scaler.fit_transform(train_data.numpy())
                X_test = scaler.transform(test_data.numpy())
                y_test = test_labels.numpy().ravel()

                # Model
                clf = KNN(n_neighbors=k)
                clf.fit(X_train)

                test_scores = clf.decision_function(X_test)
                n_anomalies = int(np.sum(y_test))
                top_indices = np.argsort(test_scores)[-n_anomalies:]

                y_pred = np.zeros_like(y_test)
                y_pred[top_indices] = 1

                results['f1'].append(f1_score(y_test, y_pred))

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Aggregate metrics
        grid_results.append({
            'scaler': s_name,
            'k': k,
            'f1_mean': np.mean(results['f1']),
            'f1_std': np.std(results['f1'])  # Added this so the print works
        })

    # --- Reporting ---
    df_grid = pd.DataFrame(grid_results)

    print("\n" + "★" * 50)
    print(f" TOP 5 CONFIGURATIONS FOR {dataset_name.upper()} ".center(50, " "))
    print("★" * 50)

    top_5 = df_grid.sort_values(by='f1_mean', ascending=False).head(5).reset_index(drop=True)

    print(top_5.to_string(index=True, formatters={
        'f1_mean': '{:.4f}'.format,
        'f1_std': '{:.4f}'.format
    }))

    print("-" * 50)
    if not top_5.empty:
        best = top_5.iloc[0]
        print(f"WINNER: Scaler={best['scaler']}, k={int(best['k'])} | F1: {best['f1_mean']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_extended_grid_search(dataset_name='wine', num_splits=50)