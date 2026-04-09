import torch
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.mcd import MCD

# Da data_loader.py im selben Ordner ist, reicht ein einfacher Import
from data_loader import Data_Loader


# --- FUNKTION 1: DIE BERECHNUNG (SINGLE RUN) ---
def run_single_mcd(test_data, test_labels):
    """
    Führt MCD auf einem Test-Set aus.
    """
    # Umwandlung Torch -> NumPy
    if torch.is_tensor(test_data):
        X_test = test_data.detach().cpu().numpy()
    else:
        X_test = test_data

    if torch.is_tensor(test_labels):
        y_test = test_labels.detach().cpu().numpy().ravel()
    else:
        y_test = np.array(test_labels).ravel()

    # Anomalie-Rate im aktuellen Split (Anomalie = 1)
    n_anomalies = np.sum(y_test == 1)
    contamination = max(0.01, n_anomalies / len(X_test))

    # Modell fitten
    clf = MCD(contamination=contamination, random_state=42)
    clf.fit(X_test)

    # Metriken berechnen
    f1 = f1_score(y_test, clf.labels_)
    f1_m = f1_score(y_test, clf.labels_, average='macro')
    auc = roc_auc_score(y_test, clf.decision_scores_)
    prc = average_precision_score(y_test, clf.decision_scores_)

    return f1, f1_m, auc, prc


# --- FUNKTION 2: DIE BENCHMARK SCHLEIFE ---
def start_mcd_benchmark(dataset_name, num_splits=100):
    dl = Data_Loader()

    # --- ABSOLUTER PFAD-FIX ---
    # Wir gehen davon aus, dass dein Projekt so aussieht:
    # Desktop/Repository/Implementation/Data/wine.mat

    # Wir bauen den Pfad von Grund auf neu:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../Implementation/Methods
    # Wir gehen zwei Ebenen hoch zum Repository-Ordner
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    # Und von dort in den Data Ordner
    mat_file_path = os.path.join(repo_root, "Reproduction", "Data", f"{dataset_name}.mat")

    all_f1, all_f1_m, all_auc, all_prc = [], [], [], []

    print(f"--- Starte MCD Benchmark für: {dataset_name} ---")
    print(f"Suche Datei unter: {mat_file_path}")  # Das hilft dir beim Debuggen!

    if not os.path.exists(mat_file_path):
        print(f"FEHLER: Datei wurde nicht gefunden! Bitte prüfe den Pfad.")
        return

    # Endergebnisse (Mean ± SD)
    print(f"\n" + "=" * 45)
    print(f"FINALE MCD ERGEBNISSE ({dataset_name})")
    print("-" * 45)
    print(f"F1 Binary: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"F1 Macro:  {np.mean(all_f1_m):.4f} ± {np.std(all_f1_m):.4f}")
    print(f"ROC-AUC:   {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"AUPRC:     {np.mean(all_prc):.4f} ± {np.std(all_prc):.4f}")
    print("=" * 45)


if __name__ == "__main__":
    # Falls wine.mat in Repository/Implementation/Data liegt
    start_mcd_benchmark('wine', num_splits=100)