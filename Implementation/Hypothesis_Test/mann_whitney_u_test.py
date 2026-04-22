import pandas as pd
from scipy.stats import mannwhitneyu

df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=None, engine='python')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=None, engine='python')
df_knn = pd.read_csv('Results/results_knn.csv',  decimal=',',sep=None, engine='python')

metrics = ['f1_score', 'roc_auc', 'auprc']

for df in [df_intercont, df_mcd, df_knn]:
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df.dropna(inplace=True)

pairs = [
    ('InterCont', 'MCD', df_intercont, df_mcd),
    ('InterCont', 'k-NN', df_intercont, df_knn),
    ('MCD', 'k-NN', df_mcd, df_knn)
]

print("\n--- PAARWEISE VERGLEICHE (Mann-Whitney-U) ---")
print("Signifikanzschwelle (Bonferroni): p < 0.0167")

for m in metrics:
    print(f"\n>>> Metrik: {m.upper()}")
    for name1, name2, data1, data2 in pairs:
        # Wir testen 'greater', um direkt zu sehen, ob die erste Methode besser ist
        stat, p_val = mannwhitneyu(data1[m], data2[m], alternative='two-sided')

        p_value = p_val.item() if hasattr(p_val, 'item') else p_val
        sig = "JA" if p_value < 0.01667 else "NEIN"

        print(f"{name1} vs {name2:5} | p-Wert: {p_value:.10f} | Signifikant: {sig}")