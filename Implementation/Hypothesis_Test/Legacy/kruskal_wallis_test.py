import pandas as pd
from scipy.stats import kruskal


df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',')
df_knn = pd.read_csv('Results/results_knn_legacy.csv', decimal=',')

metrics = ['f1_score', 'roc_auc', 'auprc']

for df in [df_intercont, df_mcd, df_knn]:
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df.dropna(inplace=True)

print("--- KRUSKAL-WALLIS-TEST ---")

for m in metrics:
    stat, p_val = kruskal(df_intercont[m], df_mcd[m], df_knn[m])

    # Sicherstellen, dass wir einen skalaren Wert zum Drucken haben
    h_stat = stat.item() if hasattr(stat, 'item') else stat
    p_value = p_val.item() if hasattr(p_val, 'item') else p_val

    print(f"\nMetrik: {m.upper()}")
    print(f"H-Statistik: {h_stat:.4f}")
    print(f"p-Wert:      {p_value:.10f}")

    # Interpretation
    if p_value < 0.05:
        print("Ergebnis: SIGNIFIKANT. Es gibt systematische Performance-Unterschiede zwischen den Modellen.")
    else:
        print("Ergebnis: NICHT SIGNIFIKANT. Keine statistisch belegbaren Unterschiede gefunden.")