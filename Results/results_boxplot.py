import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',')
df_knn = pd.read_csv('Results/results_knn.csv', decimal=',')

metrics = ['f1_score', 'roc_auc', 'auprc']

for df in [df_intercont, df_mcd, df_knn]:
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df.dropna(inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, m in enumerate(metrics):
    # Sicherstellen, dass wir die richtige Spalte aus jedem DF ziehen
    # .iloc[:, 1] wäre die zweite Spalte, aber wir nutzen den Namen zur Sicherheit:
    data_to_plot = [
        df_intercont[m],
        df_mcd[m],
        df_knn[m]
    ]

    axes[i].boxplot(data_to_plot, labels=['InterCont', 'MCD', 'k-NN'])
    axes[i].set_title(f' {m.upper()}')

    # Jetzt sollte die Achse zwischen 0 und 1 liegen
    axes[i].set_ylabel('Score')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()