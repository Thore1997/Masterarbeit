import numpy as np

# Load the file
data = np.load('Results/scores_intercont.npz')

# See what's inside (the keys)
print(data.files)

# Access a specific array using its key
array1 = data['f_1']