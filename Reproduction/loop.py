import numpy as np
import torch
from sklearn.model_selection import train_test_split
from train import trainer
import os
import scipy.io
import csv
from datetime import datetime


class Args:
    batch_size = 129
    faster_version = 'no'


args = Args()
my_trainer = trainer(args)

file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')
data = scipy.io.loadmat(file_path)
X = data['X']
y = data['y'].ravel()

log_file = "results_intercont.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Run', 'F1_Macro', 'F1_Binary', 'AUC', 'AUPRC'])

all_f1_bin = []
all_f1_macro = []
all_aucs = []
all_prc = []

# Listen für das Sammeln aller Scores über alle Runs
all_y_true = []
all_y_scores = []

# Anzahl der erfolgreichen Runs
target_successful_runs = 500
attempted_runs = 0

print(f"Starting experiment. Targeting {target_successful_runs} runs for ROC/PR data...")

normal_indices = np.where(y == 0)[0]
anomaly_indices = np.where(y == 1)[0]

while len(all_f1_macro) < target_successful_runs:
    attempted_runs += 1
    try:
        train_norm_idx, test_norm_idx = train_test_split(normal_indices, test_size=0.5)
        X_train_normal = X[train_norm_idx]

        X_test_normal_part = X[test_norm_idx]
        X_test_anomalies = X[anomaly_indices]
        X_test = np.concatenate([X_test_normal_part, X_test_anomalies], axis=0)

        y_test_normal_part = y[test_norm_idx]
        y_test_anomalies = y[anomaly_indices]
        y_test = np.concatenate([y_test_normal_part, y_test_anomalies], axis=0)

        shuffle_idx = np.random.permutation(len(y_test))
        X_test = X_test[shuffle_idx]
        y_test = y_test[shuffle_idx]
        y_test_tensor = torch.as_tensor(y_test)

        # Training und Evaluation
        f1_bin, f1_macro, auc, auprc, y_scores = my_trainer.train_and_evaluate(X_train_normal, X_test, y_test_tensor)

        # Ergebnisse an Listen anhängen
        all_f1_bin.append(f1_bin)
        all_f1_macro.append(f1_macro)
        all_aucs.append(auc)
        all_prc.append(auprc)

        # Scores und Labels für die spätere Gesamtspeicherung sammeln
        all_y_true.append(y_test)
        all_y_scores.append(y_scores)

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(all_f1_macro),
                f"{f1_macro:.4f}",
                f"{f1_bin:.4f}",
                f"{auc:.4f}",
                f"{auprc:.4f}"
            ])

        print(f"Success {len(all_f1_macro)}/{target_successful_runs} | AUC: {auc:.4f}")

    except Exception as e:
        print(f"Run {attempted_runs}: Error: {e}")
        continue

# --- Finale Speicherung aller Scores in eine einzige Datei ---
if not os.path.exists('Results'):
    os.makedirs('Results')

# Speichern als object-arrays, falls die Test-Sets in den Runs minimal variieren
np.savez('Results/scores_intercont.npz',
         y_true=np.array(all_y_true, dtype=object),
         y_scores=np.array(all_y_scores, dtype=object))
print(f"\n>>> Alle {len(all_y_scores)} Runs erfolgreich in 'Results/scores_intercont.npz' gespeichert.")

mean_f1_m, std_f1_m = np.mean(all_f1_macro), np.std(all_f1_macro)
mean_f1_b, std_f1_b = np.mean(all_f1_bin), np.std(all_f1_bin)
mean_auc, std_auc = np.mean(all_aucs), np.std(all_aucs)
mean_auprc, std_auprc = np.mean(all_prc), np.std(all_prc)

with open(log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(['SUMMARY', 'Mean', 'Std Dev'])
    writer.writerow(['F1 Macro', f"{mean_f1_m:.4f}", f"{std_f1_m:.4f}"])
    writer.writerow(['F1 Binary', f"{mean_f1_b:.4f}", f"{std_f1_b:.4f}"])
    writer.writerow(['AUC', f"{mean_auc:.4f}", f"{std_auc:.4f}"])
    writer.writerow(['AUPRC', f"{mean_auprc:.4f}", f"{std_auprc:.4f}"])

print(f"\nFinal AUC: {mean_auc:.4f} ± {std_auc:.4f}")

print("\n" + "=" * 40)
print("--- FINAL EXPERIMENT RESULTS (Semi-Supervised) ---")
print(f"Runs attempted: {attempted_runs}")
print(f"Successful runs: {len(all_f1_macro)}")
print(f"F1 Macro (Mean ± SD):  {mean_f1_m:.4f} ± {std_f1_m:.4f}")
print(f"F1 Binary (Mean ± SD): {mean_f1_b:.4f} ± {std_f1_b:.4f}")
print(f"AUPRC Score (Mean ± SD): {mean_auprc:.4f} ± {std_auprc:.4f}")
print(f"AUC (Mean ± SD):       {mean_auc:.4f} ± {std_auc:.4f}")
print("=" * 40)