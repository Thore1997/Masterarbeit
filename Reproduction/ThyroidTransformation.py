import scipy.io
import pandas as pd # You might need to run: pip install pandas

# Load the file
mat = scipy.io.loadmat('data/thyroid.mat')

# Create a DataFrame from the feature matrix X
df = pd.DataFrame(mat['X'])

# Add the labels (y) as a final column named 'Target'
df['Target'] = mat['y']

# Display the first 5 rows in a clean table format
print("Top 5 rows of the Thyroid dataset:")
print(df.head())