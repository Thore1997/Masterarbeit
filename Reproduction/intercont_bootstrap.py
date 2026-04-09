import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def run_icl_bootstrapping(final_scores, labels, B=1000):
    """
    Führt Bootstrapping auf den bereits berechneten ICL-Scores durch.

    final_scores: Das 5. Rückgabeelement aus train_and_evaluate (NumPy Array)
    labels: Die echten Kategorien (0 = Anomalie, 1 = Normal)
    B: Anzahl der Bootstrapping-Iterationen
    """

    # Metriken-Speicher
    boot_metrics = {
        'f1': [],
        'auc': [],
        'auprc': []
    }

    n_samples = len(labels)
    # Ground Truth: Im Paper-Setup ist 0 oft die Anomalie
    is_anomaly = (np.array(labels) == 0)

    print(f"Starte ICL Bootstrapping für {B} Durchläufe...")

    for i in range(B):
        # 1. Resampling: Ziehe Indizes mit Zurücklegen
        indices = np.random.choice(n_samples, n_samples, replace=True)

        boot_scores = final_scores[indices]
        boot_labels = is_anomaly[indices]

        # Falls eine Stichprobe zufällig keine Anomalien enthält, überspringen
        if np.sum(boot_labels) == 0:
            continue

        # --- ROC-AUC & AUPRC ---
        boot_metrics['auc'].append(roc_auc_score(boot_labels, boot_scores))
        boot_metrics['auprc'].append(average_precision_score(boot_labels, boot_scores))

        # --- F1-Score (Paper Protokoll) ---
        # Schwellenwert so setzen, dass Top-N Scores als Anomalie gelten
        # wobei N = Anzahl echter Anomalien in dieser Boot-Stichprobe
        n_anom = np.sum(boot_labels)
        threshold = np.sort(boot_scores)[-n_anom]
        preds = (boot_scores >= threshold).astype(int)

        boot_metrics['f1'].append(f1_score(boot_labels, preds))

    # --- Auswertung ---
    print("\n" + "=" * 30)
    print(f"ICL BOOTSTRAP ERGEBNISSE")
    print("=" * 30)
    for m in ['f1', 'auc', 'auprc']:
        mean = np.mean(boot_metrics[m])
        sd = np.std(boot_metrics[m])
        print(f"{m.upper():6}: {mean:.4f} ± {sd:.4f}")
    print("=" * 30)

    return boot_metrics