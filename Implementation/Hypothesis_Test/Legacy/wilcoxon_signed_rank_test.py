import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# 1. Dateien laden (Pfade anpassen)
df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=';', engine='python')
df_knn = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=';', engine='python')
df_mcd = pd.read_csv('Results/results_knn.csv', decimal=',', sep=';', engine='python')

metrics = ['f1_score', 'roc_auc', 'auprc']
methods = ['intercont', 'knn', 'mcd']
dfs = [df_intercont, df_knn, df_mcd]

for metrics in metrics:
    print(f"\n=== analyse {metrics.upper()} ===")

    # Daten für die aktuelle Metrik extrahieren
    data_a = df_intercont[metrics]
    data_b = df_knn[metrics]
    data_c = df_mcd[metrics]

    # 2. Friedman-Test (Globaler Unterschied)
    stat_f, p_f = friedmanchisquare(data_a, data_b, data_c)
    print(f"Friedman-Test: p = {p_f:.4e}")

    if p_f < 0.05:
        # 3. Wilcoxon Paarvergleiche (Post-hoc)
        alpha_bonf = 0.05 / 3
        print(f"Post-hoc (Alpha korrigiert: {alpha_bonf:.4f}):")

        for (d1, name1), (d2, name2) in combinations(zip(dfs, methods), 2):
            stat_w, p_w = wilcoxon(d1[metrics], d2[metrics])
            sig = "Yes" if p_w < alpha_bonf else "No"
            print(f"  {name1} vs {name2}: p = {p_w:.4e} (Significant: {sig})")
    else:
        print("Kein signifikanter Unterschied im Friedman-Test.")