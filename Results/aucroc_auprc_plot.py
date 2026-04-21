import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler

# Konfiguration
methods = {
    'InterCont': 'Results/scores_intercont.npz',
    'MCD': 'Results/scores_mcd.npz',
    'k-NN': 'Results/scores_knn.npz'
}
colors = {'InterCont': '#1f77b4', 'MCD': '#ff7f0e', 'k-NN': '#2ca02c'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for name, path in methods.items():
    if not os.path.exists(path):
        continue

    data = np.load(path)
    y_true = data['y_true']
    y_scores = data['y_scores'].reshape(-1, 1)

    # Normalisierung auf 0-1 Bereich
    scaler = MinMaxScaler()
    y_scores_norm = scaler.fit_transform(y_scores).ravel()

    # Logik-Check: Wenn Anomalien (1) im Schnitt niedrigere Scores haben als Normale (0),
    # müssen wir die Scores invertieren, damit Hoch = Anomalie gilt.
    if np.mean(y_scores_norm[y_true == 1]) < np.mean(y_scores_norm[y_true == 0]):
        y_scores_norm = 1 - y_scores_norm

    # --- 1. ROC-Kurve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores_norm)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=colors[name], lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

    # --- 2. PR-Kurve ---
    precision, recall, _ = precision_recall_curve(y_true, y_scores_norm)
    # WICHTIG: Precision-Recall Kurven müssen bei (Recall=0, Precision=1) starten oder enden
    avg_prec = average_precision_score(y_true, y_scores_norm)
    ax2.plot(recall, precision, color=colors[name], lw=2.5, label=f'{name} (AP = {avg_prec:.3f})')

# Styling ROC
ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC-Kurven Vergleich', fontsize=14, fontweight='bold')
ax1.legend(loc="lower right")
ax1.grid(True, linestyle=':', alpha=0.6)

# Styling PR
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Kurven Vergleich', fontsize=14, fontweight='bold')
ax2.legend(loc="upper right") # Legende nach oben rechts, da Kurven meist fallen
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Results/final_comparison_curves.png', dpi=300)
plt.show()