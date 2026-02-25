import scipy.io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Load the data
mat = scipy.io.loadmat('Data/thyroid_processed_dataset.mat')
data = mat['X']  # Features
y = mat['Y'].ravel()  # Class labels (0 for normal, 1 for outlier)

# 2. Define your feature names (The user-provided labels)
feature_names = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

# 3. Preprocessing: Standardizing is crucial for t-SNE performance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 4. Core t-SNE Calculation
# n_components=2 for a 2D plot
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
data_embedded = tsne.fit_transform(data_scaled)

# 5. Visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_embedded[:, 0], data_embedded[:, 1],
                      c=y, cmap='coolwarm', alpha=0.7, edgecolors='k', s=20)

plt.title('t-SNE Visualization of Thyroid Dataset')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Class Label (Normal vs Anomaly)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()