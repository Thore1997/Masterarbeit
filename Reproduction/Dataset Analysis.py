import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
thyroid = sio.loadmat('Data/thyroid.mat')
X = thyroid['X']
df = pd.DataFrame(X)


def plot_boxplots_no_outliers(df, filename="thyroid_boxplots_clean.pdf"):
    numeric_cols = df.select_dtypes(include=['number']).columns
    sns.set_theme(style="whitegrid")

    n_cols = 2
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        data = df[col]

        # 1. Plot the boxplot WITHOUT outliers (showfliers=False)
        sns.boxplot(x=data, ax=axes[i], color='skyblue', showfliers=False)

        # 2. Add Mean/Median lines
        col_mean = data.mean()
        col_median = data.median()

        axes[i].axvline(col_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {col_mean:.2f}')
        axes[i].axvline(col_median, color='green', linestyle='-', linewidth=2, label=f'Median: {col_median:.2f}')

        # 3. Titles and Labels
        axes[i].set_title(f'Feature {col} ', fontsize=14)
        axes[i].legend(fontsize='small')
        axes[i].set_xlabel('Value')

    # Remove extra empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Clean boxplot report saved to {filename}")
    plt.show()


# Run the script
plot_boxplots_no_outliers(df)