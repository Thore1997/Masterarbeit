import pandas as pd
from scipy import stats

# 1. Load Data
# Using sep=None with engine='python' is great for auto-detecting ; vs ,
df_intercont = pd.read_csv('Results/results_intercont.csv', decimal=',', sep=None, engine='python')
df_mcd = pd.read_csv('Results/results_mcd.csv', decimal=',', sep=None, engine='python')
df_knn = pd.read_csv('Results/results_knn.csv', decimal=',', sep=None, engine='python')

metrics = ['f1_score', 'roc_auc', 'auprc']

print("--- GLOBAL FRIEDMAN TEST RESULTS ---")

for m in metrics:
    # Extract the data for the current metric
    # .dropna() ensures we don't pass 'NaN' values which break the test
    data_a = df_intercont[m].dropna().values
    data_b = df_knn[m].dropna().values
    data_c = df_mcd[m].dropna().values

    statistic, p_value = stats.friedmanchisquare(data_a, data_b, data_c)

    sig = "YES" if p_value < 0.05 else "NO"

    print(f"\nMetric: {m.upper()}")
    print(f"  - Friedman Stat:  {statistic:.4f}")
    print(f"  - P-value:        {p_value:.10f}")
    print(f"  - Significant?    {sig}")


