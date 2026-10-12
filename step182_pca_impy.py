import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.array([[1, 1], [2, 3], [3, 2], [4, 4]], dtype=float)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print("Scores Z:\n", np.round(Z, 4))
print("Components:\n", np.round(pca.components_, 4))
print("Explained ratio:", pca.explained_variance_ratio_)

from sklearn.decomposition import PCA

X = np.array([[20, 45, 3, 120], [25, 55, 4, 110], [18, 40, 2, 130],
              [30, 62, 6, 100], [22, 50, 5, 125], [28, 58, 3, 105]], dtype=float)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print("Z shape:", Z.shape)
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
print("Sum:", round(float(pca.explained_variance_ratio_.sum()), 4))

X = np.array([
    [20, 45, 3, 120],
    [25, 55, 4, 110],
    [18, 40, 2, 130],
    [30, 62, 6, 100],
    [22, 50, 5, 125],
    [28, 58, 3, 105]
], dtype=float)

n = X.shape[0]
X_c = X - np.mean(X, axis=0)
C = (X_c.T @ X_c) / (n - 1)

vals, vecs = np.linalg.eigh(C)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

W = vecs[:, :2]
Z_manual = X_c @ W

pca = PCA(n_components=2)
Z_sklearn = pca.fit_transform(X)

print("np.allclose(Z_manual, Z_sklearn):", np.allclose(Z_manual, Z_sklearn))
print("np.allclose(np.abs(Z_manual), np.abs(Z_sklearn)):", np.allclose(np.abs(Z_manual), np.abs(Z_sklearn)))

print("np.allclose(W.T, pca.components_):", np.allclose(W.T, pca.components_))
print("np.allclose(np.abs(W.T), np.abs(pca.components_)):", np.allclose(np.abs(W.T), np.abs(pca.components_)))

# 1. Imports (numpy, plt, PCA) and define X (8x5) with feature names

X = np.array([
    [5, 7, 3, 6, 2],
    [6, 8, 5, 4, 3],
    [4, 5, 2, 7, 1],
    [8, 9, 7, 3, 5],
    [5, 6, 6, 5, 4],
    [7, 8, 4, 2, 4],
    [3, 4, 3, 8, 2],
    [6, 7, 8, 4, 6]
], dtype=float)

feature_names = ['tasks', 'cpu', 'queue', 'idle', 'alerts']

# 2. Fit full PCA; compute individual and cumulative variance ratios
pca = PCA()
pca.fit(X)

evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

# 3. Plot and save scree.png (bars = individual, line = cumulative)
comps = np.arange(1, len(evr) + 1)

plt.figure(figsize=(8, 5))
plt.bar(comps, evr, alpha=0.6, label='Individual explained variance')
plt.plot(comps, cum_evr, marker='o', color='red', label='Cumulative explained variance')
plt.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Sentinel Telemetry Scree Plot')
plt.xticks(comps)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig('scree.png')
plt.close()

# 4. Find smallest k reaching cumulative ratio >= 0.95 (with 1e-6 tol)
tol = 1e-6
k = int(np.argmax(cum_evr >= 0.95 - tol) + 1)
print(f"Explained Variance Ratios: {np.round(evr, 4)}")
print(f"Cumulative Ratios: {np.round(cum_evr, 4)}")
print(f"Smallest k for >= 95% variance: {k}")

# 5. Extract PC1/PC2 loadings and print top feature by |loading| for each
loadings = pca.components_

print("\nPC1 Loadings:")
for name, val in zip(feature_names, loadings[0]):
    print(f"  {name:7s}: {val:+.4f}")
top_pc1 = feature_names[np.argmax(np.abs(loadings[0]))]
print(f"Top feature for PC1 by |loading|: {top_pc1} ({abs(loadings[0][feature_names.index(top_pc1)]):.4f})")

print("\nPC2 Loadings:")
for name, val in zip(feature_names, loadings[1]):
    print(f"  {name:7s}: {val:+.4f}")
top_pc2 = feature_names[np.argmax(np.abs(loadings[1]))]
print(f"Top feature for PC2 by |loading|: {top_pc2} ({abs(loadings[1][feature_names.index(top_pc2)]):.4f})")
