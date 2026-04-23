import torch
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD
from sklearn.preprocessing import RobustScaler
from data_loader import Data_Loader


def run_mcd_top10_benchmark(dataset_name, num_splits=500, top_k=10):
    dl = Data_Loader()

    # Pfad-Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    if not os.path.exists(mat_file_path):
        print(f"FEHLER: Datei nicht gefunden: {mat_file_path}")
        return

    all_f1, all_auc, all_prc = [], [], []
    print(f"--- MCD Benchmark ({dataset_name}) | Top-{top_k} als Anomalien ---")

    for i in range(num_splits):
        try:
            # 1. Daten laden (Semi-supervised: train_data enthält meist nur Normale)
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(mat_file_path)

            # Konvertierung zu NumPy
            X_train = train_data.detach().cpu().numpy() if torch.is_tensor(train_data) else train_data
            X_test = test_data.detach().cpu().numpy() if torch.is_tensor(test_data) else test_data
            y_test = test_labels.detach().cpu().numpy().ravel() if torch.is_tensor(test_labels) else np.array(
                test_labels).ravel()

            # 2. Skalierung
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 3. MCD Training (berechnet Mean und Covariance der sauberen Daten)
            # Contamination klein halten, da wir davon ausgehen, dass Train sauber ist
            clf = MCD(contamination=0.0001, random_state=42)
            clf.fit(X_train_scaled)

            # 4. Scoring (Mahalanobis-Distanz auf Testdaten anwenden)
            scores = clf.decision_function(X_test_scaled)

            # 5. Top-K Logik: Die 10 weitesten Entfernungen markieren
            pred_labels = np.zeros(len(scores))
            top_k_indices = np.argsort(scores)[-top_k:]  # Holt die Indizes der höchsten Scores
            pred_labels[top_k_indices] = 1

            # 6. Metriken speichern
            all_f1.append(f1_score(y_test, pred_labels))
            all_auc.append(roc_auc_score(y_test, scores))
            all_prc.append(average_precision_score(y_test, scores))

            if (i + 1) % 50 == 0:
                print(f"Fortschritt: {i + 1}/{num_splits} Splits berechnet...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # Finale Statistik
    if all_f1:
        print(f"\n" + "=" * 45)
        print(f"ERGEBNISSE FÜR {dataset_name.upper()} (Top-{top_k} Ansatz)")
        print("-" * 45)
        print(f"F1-Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        print(f"ROC-AUC:   {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
        print(f"AUPRC:     {np.mean(all_prc):.4f} ± {np.std(all_prc):.4f}")
        print("=" * 45)


if __name__ == "__main__":
    # Hier kannst du den Namen und die Anzahl der Top-Anomalien anpassen
    run_mcd_top10_benchmark('wine', num_splits=500, top_k=10)

    #TODO:  csv und npz  export noch einfügen