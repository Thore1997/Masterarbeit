import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten laden
df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',')
df_knn = pd.read_csv('Results/results_knn.csv', decimal=',')

metriken = ['f1_score', 'roc_auc', 'auprc']
colors = {"InterCont": "blue", "MCD": "orange", "k-NN": "green"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, m in enumerate(metriken):
    # WICHTIG: Hier runden wir auf 2 Nachkommastellen
    d1 = pd.to_numeric(df_intercont[m], errors='coerce').round(2).dropna()
    d2 = pd.to_numeric(df_mcd[m], errors='coerce').round(2).dropna()
    d3 = pd.to_numeric(df_knn[m], errors='coerce').round(2).dropna()

    # ECDF Plots zeichnen
    sns.ecdfplot(d1, ax=axes[i], label="InterCont", color=colors["InterCont"], lw=2.5)
    sns.ecdfplot(d2, ax=axes[i], label="MCD", color=colors["MCD"], lw=2.5)
    sns.ecdfplot(d3, ax=axes[i], label="k-NN", color=colors["k-NN"], lw=2.5)

    axes[i].set_title(f"Cumulative Distribution: {m.upper()}")
    axes[i].set_xlabel("Score (Rouded to 0.01)")
    axes[i].set_ylabel("Share of Runs")
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].legend()

plt.tight_layout()
plt.show()