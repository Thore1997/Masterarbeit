import scipy.io
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler


mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')
data = mat['X']
y = mat['y'].ravel()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. UMAP Implementation
# n_neighbors is roughly equivalent to t-SNE's perplexity
# min_dist controls how tightly UMAP packs points together (0.1 is standard)
reducer = umap.UMAP(n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42)
data_embedded = reducer.fit_transform(data_scaled)


plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_embedded[:, 0], data_embedded[:, 1],
                      c=y, cmap='coolwarm', alpha=0.7, edgecolors='k', s=20)

plt.title('UMAP Visualization of Wine Dataset')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(scatter, label='Class Label')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()