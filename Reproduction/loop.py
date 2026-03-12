import numpy as np
import torch
from sklearn.model_selection import train_test_split
from train import trainer  # Imports your trainer class
import os
import scipy


# 1. Setup Data and Hyperparameters
class Args:
    batch_size = 3000
    faster_version = 'no'

args = Args()
my_trainer = trainer(args)

# Load the data
file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')
data = scipy.io.loadmat(file_path)
X = data['X']
y = data['y'].ravel()

all_f1s = []
all_aucs = []
all_prc = []

target_successful_runs = 500
attempted_runs = 0

print(f"Starting experiment. Targeting {target_successful_runs} manual splits...")

# Identify locations of classes once before the loop
normal_indices = np.where(y == 0)[0]
anomaly_indices = np.where(y == 1)[0]

# 2. The Experimental Loop
while len(all_f1s) < target_successful_runs:
    attempted_runs += 1

    try:
        # 1. Split ONLY the normal data into 50% train and 50% test
        train_norm_idx, test_norm_idx = train_test_split(normal_indices, test_size=0.5, random_state=42)

        # 2. Create Training Set
        X_train_normal = X[train_norm_idx]

        # 3. Create Test Set (Rest of Normals + ALL Anomalies)
        X_test_normal_part = X[test_norm_idx]
        X_test_anomalies = X[anomaly_indices]
        X_test = np.concatenate([X_test_normal_part, X_test_anomalies], axis=0)

        # 4. Create Test Labels
        y_test_normal_part = y[test_norm_idx]
        y_test_anomalies = y[anomaly_indices]
        y_test = np.concatenate([y_test_normal_part, y_test_anomalies], axis=0)

        # Shuffle X_test and y_test together so anomalies aren't always at the end
        shuffle_idx = np.random.permutation(len(y_test))
        X_test = X_test[shuffle_idx]
        y_test = y_test[shuffle_idx]

        y_test_tensor = torch.as_tensor(y_test)

        # 5. Pass to the trainer
        f1, auc, auprc = my_trainer.train_and_evaluate(X_train_normal, X_test, y_test_tensor)

        all_f1s.append(f1)
        all_aucs.append(auc)
        all_prc.append(auprc)

        print(f"Success {len(all_f1s)}/{target_successful_runs} | F1: {f1:.4f} | AUC: {auc:.4f} | AUPRC: {auprc:.4f}")

    except Exception as e:
        print(f"Run {attempted_runs}: Error encountered: {e}")
        continue

# 3. Calculate Final Statistics
mean_f1 = np.mean(all_f1s)
std_f1 = np.std(all_f1s)
mean_auc = np.mean(all_aucs)
std_auc = np.std(all_aucs)
mean_auprc = np.mean(all_prc)
std_auprc = np.std(all_prc)

print("\n" + "=" * 40)
print("--- FINAL EXPERIMENT RESULTS (Semi-Supervised) ---")
print(f"Runs attempted: {attempted_runs}")
print(f"Successful runs: {len(all_f1s)}")
print(f"F1 Score (Mean ± SD): {mean_f1:.4f} ± {std_f1:.4f}")
print(f"AUPRC Score (Mean ± SD): {mean_auprc:.4f} ± {std_auprc:.4f}")
print(f"AUC (Mean ± SD):{mean_auc:.4f} ± {std_auc:.4f}")
print("=" * 40)