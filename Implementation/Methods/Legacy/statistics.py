import numpy as np
from scipy import stats


def perform_summary_anova(metric_name, means, sds, ns):
    """
    Calculates One-Way ANOVA from summary statistics.
    """
    k = len(means)
    n_total = sum(ns)

    # 1. Calculate Grand Mean
    grand_mean = sum(m * n for m, n in zip(means, ns)) / n_total

    # 2. Sum of Squares Between (SSB)
    ssb = sum(n * (m - grand_mean) ** 2 for m, n in zip(means, ns))

    # 3. Sum of Squares Within (SSW)
    ssw = sum((n - 1) * (sd ** 2) for sd, n in zip(sds, ns))

    # 4. Mean Squares
    df_between = k - 1
    df_within = n_total - k
    ms_between = ssb / df_between
    ms_within = ssw / df_within

    # 5. F-statistic and P-value
    f_stat = ms_between / ms_within
    p_value = stats.f.sf(f_stat, df_between, df_within)

    return {
        "metric": metric_name,
        "f_stat": f_stat,
        "p_value": p_value,
        "df": (df_between, df_within)
    }


# --- 1. DEFINE YOUR DATA ---
# Replace the 'm' and 's' values with your actual results.
# Format: [Method_1, Method_2, Method_3]
data_package = {
    "F1-Score": {
        "m": [0.6414, 0.6812, 0.7744],
        "s": [0.0941, 0.0862, 0.1059],
        "n": [500, 500, 500]
    },
    "ROC AUC": {
        "m": [0.9594, 0.9587, 0.9670],
        "s": [0.0359, 0.0189, 0.0206],
        "n": [500, 500, 500]
    },
    "AUPRC": {
        "m": [0.7782, 0.7285, 0.9948],
        "s": [0.1643, 0.1154, 0.032],
        "n": [500, 500, 500]
    }
}

# --- 2. LOOP THROUGH METRICS ---
print(f"{'METRIC':<12} | {'F-STAT':<10} | {'P-VALUE':<12} | {'RESULT'}")
print("-" * 60)

for metric, vals in data_package.items():
    res = perform_summary_anova(metric, vals['m'], vals['s'], vals['n'])

    status = "SIGNIFICANT" if res['p_value'] < 0.05 else "not sig."

    print(f"{res['metric']:<12} | {res['f_stat']:<10.4f} | {res['p_value']:<12.2e} | {status}")

    # --- 3. OPTIONAL: PAIRWISE POST-HOC (Only if ANOVA is significant) ---
    if res['p_value'] < 0.0167:
        print(f"   > Post-hoc pairwise comparisons for {metric}:")
        pairs = [(0, 1), (1, 2), (0, 2)]
        method_names = ["MCD", "k-NN", "InterCont"]

        for i, j in pairs:
            # Welch's T-Test from stats
            t_stat, p_pair = stats.ttest_ind_from_stats(
                mean1=vals['m'][i], std1=vals['s'][i], nobs1=vals['n'][i],
                mean2=vals['m'][j], std2=vals['s'][j], nobs2=vals['n'][j],
                equal_var=False
            )
            # Apply Bonferroni correction threshold (0.05 / 3 = 0.0167)
            sig_pair = "*" if p_pair < (0.05 / 3) else " "
            print(f"     - {method_names[i]} vs {method_names[j]}: p = {p_pair:.4e} {sig_pair}")
    print("-" * 60)