import torch
import numpy as np
import os
import scipy.io
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD
from sklearn.preprocessing import RobustScaler
from data_loader import Data_Loader


# --- FUNKTION 1: DIE BERECHNUNG (KORRIGIERT) ---
def run_single_mcd(train_data, test_data, test_labels, contamination=0.077):
    # Umwandlung Torch -> NumPy
    X_train = train_data.detach().cpu().numpy() if torch.is_tensor(train_data) else train_data
    X_test = test_data.detach().cpu().numpy() if torch.is_tensor(test_data) else test_data
    y_test = test_labels.detach().cpu().numpy().ravel() if torch.is_tensor(test_labels) else np.array(
        test_labels).ravel()

    # 1. Skalierung (Fit auf Train, Transform auf Test)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. MCD Modell (Fit auf sauberen Trainingsdaten)
    # Da train_data nur Normale enthält, setzen wir eine kleine contamination
    clf = MCD(contamination=0.154, random_state=42)
    clf.fit(X_train_scaled)

    # 3. Ergebnisse auf Test-Set
    scores = clf.decision_function(X_test_scaled)
    labels = clf.predict(X_test_scaled)

    # 4. Auswertung
    auc = roc_auc_score(y_test, scores)
    prc = average_precision_score(y_test, scores)
    f1 = f1_score(y_test, labels)

    return f1, auc, prc


def start_mcd_benchmark(dataset_name, num_splits=500):
    dl = Data_Loader()

    # Pfad-Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    all_f1, all_auc, all_prc = [], [], []

    print(f"--- Starte Semi-Supervised MCD Benchmark: {dataset_name} ---")

    if not os.path.exists(mat_file_path):
        print(f"FEHLER: Datei nicht gefunden: {mat_file_path}")
        return

    for i in range(num_splits):
        try:
            # 1. Split holen
            train_data, test_data, test_labels = dl.build_train_test_generic_matfile(mat_file_path)

            # 2. Funktion mit den DATEN aufrufen (nicht mit dem Namen)
            f1, auc, prc = run_single_mcd(train_data, test_data, test_labels)

            all_f1.append(f1)
            all_auc.append(auc)
            all_prc.append(prc)

            if (i + 1) % 50 == 0:
                print(f"Split {i + 1}/{num_splits} fertig...")

        except Exception as e:
            print(f"Fehler in Split {i}: {e}")

    # --- FINALE AUSGABE ---
    if len(all_f1) > 0:
        print(f"\n" + "=" * 45)
        print(f"FINALE MCD ERGEBNISSE ({dataset_name})")
        print("-" * 45)
        print(f"F1 Binary: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        print(f"ROC-AUC:   {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
        print(f"AUPRC:     {np.mean(all_prc):.4f} ± {np.std(all_prc):.4f}")
        print("=" * 45)


if __name__ == "__main__":
    start_mcd_benchmark('wine', num_splits=500)