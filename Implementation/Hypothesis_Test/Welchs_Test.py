import scipy.stats as stats


def compare_results(name, my_mean, my_sd, my_n, paper_mean, paper_sd, paper_n):
    # Perform Welch's T-test (equal_var=False)
    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=my_mean, std1=my_sd, nobs1=my_n,
        mean2=paper_mean, std2=paper_sd, nobs2=paper_n,
        equal_var=False  # This makes it Welch's T-test
    )

    print(f"--- Results for {name} ---")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_value:.4f}")

    if p_value < 0.05:
        print("Conclusion:  Statistically SIGNIFICANT difference (p < 0.05)")
    else:
        print("Conclusion:  No significant difference (p >= 0.05)")
    print("\n")


# --- INPUT YOUR DATA HERE ---
# Example: F1 Score
# compare_results(metric_name, mean, sd, n_folds, paper_mean, paper_sd, paper_n)

compare_results("F1 Score", 0.8554, 0.075, 500, 0.9, 0.063, 500)
compare_results("ROC AUC", 0.989, 0.0109, 500, 0.995, 0.6, 500)
