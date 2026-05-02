import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD
from sklearn.preprocessing import RobustScaler
from data_loader import Data_Loader


def run_mcd_unsupervised_topk_benchmark(dataset_name, num_splits=500, top_k=10):
    dl = Data_Loader()

    # Pfad-Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(mat_file_path):
        print(f"FEHLER: Datei nicht gefunden: {mat_file_path}")
        return

    run_results = []
    all_scores_list = []
    all_labels_list = []

    print(f"--- MCD Unsupervised Benchmark ({dataset_name}) | Top-{top_k} Ansatz ---")

    for i in range(num_splits):
        try:
            # 1. Daten laden
            _, test_data, test_labels = dl.build_train_test_generic_matfile(mat_file_path)

            X_test = test_data.detach().cpu().numpy() if torch.is_tensor(test_data) else test_data
            y_test = test_labels.detach().cpu().numpy().ravel() if torch.is_tensor(test_labels) else np.array(
                test_labels).ravel()

            # 2. Skalierung
            scaler = RobustScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            # 3. MCD Training
            clf = MCD(contamination=0.17, random_state=42)
            clf.fit(X_test_scaled)

            # 4. Scoring
            scores = clf.decision_scores_

            # 5. Top-K Logik
            pred_labels_topk = np.zeros(len(scores))
            top_k_indices = np.argsort(scores)[-top_k:]
            pred_labels_topk[top_k_indices] = 1

            # 6. Metriken berechnen
            f1 = f1_score(y_test, pred_labels_topk)
            auc = roc_auc_score(y_test, scores)
            prc = average_precision_score(y_test, scores)

            # Speichern für CSV
            run_results.append({
                "run": i + 1,
                "f1_score": f1,
                "roc_auc": auc,
                "auprc": prc
            })

            # Speichern für NPZ
            all_scores_list.append(scores)
            all_labels_list.append(y_test)

            if (i + 1) % 50 == 0:
                print(f"Fortschritt: {i + 1}/{num_splits} Splits berechnet...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # --- EXPORT LOGIK ---
    if run_results:
        # 1. CSV Export (Metriken pro Split)
        df = pd.DataFrame(run_results)
        csv_path = os.path.join(current_dir, f"mcd_top{top_k}_{dataset_name}_metrics.csv")
        df.to_csv(csv_path, index=False)

        # 2. NPZ Export (Konsistent zu k-NN für Vergleich)
        if not os.path.exists('Results'):
            os.makedirs('Results')

        final_y_true = np.concatenate(all_labels_list)
        final_y_scores = np.concatenate(all_scores_list)

        npz_path = 'Implementation/scores_mcd.npz'
        np.savez(npz_path, y_true=final_y_true, y_scores=final_y_scores)

        print(f"\nDATEIEN GESPEICHERT:")
        print(f"1. CSV: {csv_path}")
        print(f"2. NPZ: {npz_path}")

        # Finale Statistik
        print(f"\n" + "=" * 45)
        print(f"ZUSAMMENFASSUNG: {dataset_name.upper()} (Top-{top_k})")
        print("-" * 45)
        print(f"F1-Score:  {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
        print(f"ROC-AUC:   {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}")
        print(f"AUPRC:     {df['auprc'].mean():.4f} ± {df['auprc'].std():.4f}")
        print("=" * 45)


if __name__ == "__main__":
    run_mcd_unsupervised_topk_benchmark('wine', num_splits=500, top_k=10)