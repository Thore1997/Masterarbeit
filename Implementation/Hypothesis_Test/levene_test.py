import pandas as pd
from scipy.stats import levene

# 1. Daten einlesen
df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=None, engine='python')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=None, engine='python')
df_knn = pd.read_csv('Results/results_knn.csv',  decimal=',',sep=None, engine='python')

# --- DATEN-REINIGUNG (Behebt den TypeError) ---
metrics = ['f1_score', 'roc_auc', 'auprc']

for df in [df_intercont, df_mcd, df_knn]:
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df.dropna(inplace=True)

# --- DEIN LEVENE-TEST CODE (UNVERÄNDERT) ---
for m in metrics:
    print(f"\n=== Levene-Test für: {m} ===")

    # Daten für die aktuelle Metrik extrahieren & säubern
    try:
        # Test durchführen
        stat, p_val = levene(df_intercont[m], df_mcd[m], df_knn[m], center='median')

        # Sicherer Print (falls stat ein Array ist)
        s_value = stat[0] if hasattr(stat, "__len__") else stat
        p_value = p_val[0] if hasattr(p_val, "__len__") else p_val

        print(f"Statistik: {float(s_value):.4f}")
        print(f"p-Wert:    {float(p_value):.4f}")

        if p_value < 0.05:
            print(f"ERGEBNIS: Varianzen bei {m} UNGLEICH (Heteroskedastizität)")
        else:
            print(f"ERGEBNIS: Varianzen bei {m} GLEICH (Homoskedastizität)")

    except KeyError:
        print(f"Fehler: Metrik '{m}' wurde in der CSV nicht gefunden!")