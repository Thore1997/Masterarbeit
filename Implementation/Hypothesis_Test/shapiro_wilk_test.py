import scipy.io
import pandas as pd
from scipy import stats


# Load the .mat file
mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')

# Check the keys to identify your variable name
# (MATLAB files often contain metadata like '__header__', '__version__')
print(mat.keys())

# Replace 'your_variable_name' with the actual key from the printout above
data_array = mat['X']

# Convert to DataFrame (assuming 129 rows, 13 columns)
df = pd.DataFrame(data_array)
# 1. Create a dictionary to store results
results = {}

for column in df.columns:
    stat, p_value = stats.shapiro(df[column])
    # A dimension is "Normal" if p_value > 0.05
    results[column] = {'Statistic': stat, 'p-value': p_value, 'Normal': p_value > 0.05}

# 2. Convert to DataFrame for a clean view
shapiro_results = pd.DataFrame(results).T
print(shapiro_results)