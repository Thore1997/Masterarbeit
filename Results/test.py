import numpy as np

# Load the file
data = np.load('Results/scores_mcd.npz')

# Print the names of the keys
print("Keys in this file:", data.files)

# It's good practice to close it afterward
data.close()