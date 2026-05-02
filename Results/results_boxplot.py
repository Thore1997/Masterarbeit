import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=None, engine='python')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=None, engine='python')
df_knn = pd.read_csv('Results/results_knn.csv', decimal=',', sep=None, engine='python')

metrics = ['f1_score', 'roc_auc', 'auprc']

for df in [df_intercont, df_mcd, df_knn]:
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df.dropna(inplace=True)

# Define the colors for the three methods
colors = ['#AEC6CF', '#FFB347', '#77DD77']

fig, axes = plt.subplots(1, 3, figsize=(18, 9))

for i, m in enumerate(metrics):
    data_to_plot = [
        df_intercont[m],
        df_mcd[m],
        df_knn[m]
    ]

    # patch_artist=True is required to fill the boxes with color
    bp = axes[i].boxplot(data_to_plot,
                         labels=['InterCont', 'MCD', 'k-NN'],
                         patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 2})

    # Loop through the boxes and apply the colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[i].set_title(f' {m.upper()}')
    axes[i].set_ylabel('Score')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()