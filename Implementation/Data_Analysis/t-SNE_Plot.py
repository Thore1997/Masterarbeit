import scipy.io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


mat = scipy.io.loadmat('../../Reproduction/Data/wineori.mat')
data = mat['X']
y = mat['y'].ravel()


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


tsne = TSNE(n_components=2, perplexity=100, n_iter=1000, random_state=42)
data_embedded = tsne.fit_transform(data_scaled)


plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_embedded[:, 0], data_embedded[:, 1],
                      c=y, cmap='coolwarm', alpha=0.7, edgecolors='k', s=20)

plt.title('t-SNE Visualization of Thyroid Dataset')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Class Label (Normal vs Anomaly)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()