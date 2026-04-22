import torch
import numpy as np
import os
import scipy.io
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD
from sklearn.preprocessing import RobustScaler
from data_loader import Data_Loader


def run_single_mcd(train_data, test_data, test_labels, contamination=0.077):
    X_train = train_data.detach().cpu().numpy() if torch.is_tensor(train_data) else train_data
    X_test = test_data.detach().cpu().numpy() if torch.is_tensor(test_data) else test_data
    y_test = test_labels.detach().cpu().numpy().ravel() if torch.is_tensor(test_labels) else np.array(
        test_labels).ravel()

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MCD(contamination=0.154, random_state=42)
    clf.fit(X_train_scaled)

    scores = clf.decision_function(X_test_scaled)
    labels = clf.predict(X_test_scaled)

    auc = roc_auc_score(y_test, scores)
    prc = average_precision_score(y_test, scores)
    f1 = f1_score(y_test, labels)

    return f1, auc, prc, scores, y_test


def start_mcd_benchmark(dataset_name, num_splits=1):
    dl = Data_Loader()

    # Absoluten Pfad erzwingen
    base_path = os.getcwd()
    results_dir = os.path.join(base_path, 'Results')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Ordner erstellt: {results_dir}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    all_f1, all_auc, all_prc = [], [], []

    # Listen zum Sammeln ALLER Vorhersagen für den globalen Plot
    all_y_true_collected = []
    all_y_scores_collected = []

    print(f"--- Starte MCD Benchmark: {dataset_name} ---")

    for i in range(num_splits):
        try:
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(mat_file_path)
            f1, auc, prc, scores, y_test_vals = run_single_mcd(train_data, test_data, test_labels)

            # Sammle Ergebnisse in den globalen Listen
            all_y_true_collected.append(y_test_vals)
            all_y_scores_collected.append(scores)

            all_f1.append(f1)
            all_auc.append(auc)
            all_prc.append(prc)

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"KRITISCHER FEHLER im Loop bei Split {i}: {e}")

    # --- DER EXPORT (Nach dem Loop, alle Daten kombiniert) ---
    if len(all_y_true_collected) > 0:
        save_path = os.path.join(results_dir, 'scores_mcd.npz')

        final_y_true = np.concatenate(all_y_true_collected)
        final_y_scores = np.concatenate(all_y_scores_collected)

        np.savez(save_path, y_true=final_y_true, y_scores=final_y_scores)

        if os.path.exists(save_path):
            print(f"\n!!! ERFOLG: Global MCD Scores ({len(final_y_true)} Samples) gespeichert: {save_path}")

    # --- CSV Export ---
    if len(all_f1) > 0:
        results_data = {'f1_score': all_f1, 'roc_auc': all_auc, 'auprc': all_prc}
        df = pd.DataFrame(results_data)
        csv_path = f"mcd_results_{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Statistiken in {csv_path} gespeichert.")


if __name__ == "__main__":
    start_mcd_benchmark('wine', num_splits=500)