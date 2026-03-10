from scipy import stats
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
thyroid = sio.loadmat('../../Reproduction/Data/Thyroid/thyroid.mat')
X = thyroid['X']
df = pd.DataFrame(X)


def test_normality(df):
    results = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stat, p_value = stats.shapiro(df[col].dropna())
            results.append({
                'Column Index': col,
                'W-Statistic': round(stat, 4),
                'p-value': f"{p_value:.4e}"
            })
    return pd.DataFrame(results)


def export_normality_report_with_h0(report_df, filename="Normality_Report.pdf"):
    # 1. Exclude the last column (if it was 'Normal?')
    # In the current 'results' list, we already excluded it,
    # but this line ensures only the first 3 columns are used regardless.
    report_df = report_df.iloc[:, :3]

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')

    # 3. Create the table
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=["#e6e6e6"] * len(report_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 4. Implement H0 into the Header/Title
    header_text = (
            "Shapiro-Wilk Normality Test\n"
            r"$H_0$: The data is drawn from a Normal Distribution" + "\n"
                                                                     "(Reject $H_0$ if p-value < 0.05)"
    )

    plt.title(header_text, fontsize=12, pad=30, fontweight='bold', linespacing=1.5)

    # Save to PDF
    plt.savefig(filename, bbox_inches='tight')
    print(f"Report with H0 header exported to {filename}")
    plt.show()


# Execution
normality_results = test_normality(df)
export_normality_report_with_h0(normality_results)