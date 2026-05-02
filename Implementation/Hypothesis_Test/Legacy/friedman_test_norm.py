import pandas as pd
from scipy import stats

# 1. Load Data
# Using sep=None with engine='python' is great for auto-detecting ; vs ,
df_minmax = pd.read_csv('Results/knn_results_minmax.csv', decimal=',', sep=None, engine='python')
df_stan = pd.read_csv('Results/knn_results_stan.csv', decimal=',', sep=None, engine='python')
df_robust = pd.read_csv('Results/results_knn.csv', decimal=',', sep=None, engine='python')

metrics = ['f1_score']

print("--- GLOBAL FRIEDMAN TEST RESULTS ---")

for m in metrics:
    # Extract the data for the current metric
    # .dropna() ensures we don't pass 'NaN' values which break the test
    data_a = df_minmax[m].dropna().values
    data_b = df_stan[m].dropna().values
    data_c = df_robust[m].dropna().values

    statistic, p_value = stats.friedmanchisquare(data_a, data_b, data_c)

    sig = "YES" if p_value < 0.05 else "NO"

    print(f"\nMetric: {m.upper()}")
    print(f"  - Friedman Stat:  {statistic:.4f}")
    print(f"  - P-value:        {p_value:.10f}")
    print(f"  - Significant?    {sig}")


