import pandas as pd
from scipy import stats

# 1. Load Data
df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=None, engine='python')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=None, engine='python')
df_knn = pd.read_csv('Results/results_knn.csv', decimal=',', sep=None, engine='python')

metrics = ['f1_score', 'roc_auc', 'auprc']

print("--- BRUNNER-MUNZEL TEST RESULTS ---")

for m in metrics:
    # Daten extrahieren
    data_intercont = df_intercont[m].dropna().values
    data_knn = df_knn[m].dropna().values
    data_mcd = df_mcd[m].dropna().values

    # Vergleiche definieren (Paarweise)
    comparisons = [
        ("Intercont vs KNN", data_intercont, data_knn),
        ("Intercont vs MCD", data_intercont, data_mcd),
        ("KNN vs MCD", data_knn, data_mcd)
    ]

    print(f"\nMETRIC: {m.upper()}")
    print("-" * 30)

    for label, group1, group2, in comparisons:
        # Brunner-Munzel Test
        # alternative='two-sided' ist Standard
        statistic, p_value = stats.brunnermunzel(group1, group2)

        sig = "YES" if p_value < 0.0055 else "NO"

        print(f"{label}:")
        print(f"  - BM Statistic: {statistic:.4f}")
        print(f"  - P-value:      {p_value:.10f}")
        print(f"  - Significant?  {sig}")