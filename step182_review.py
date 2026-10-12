import numpy as np

# 1. fit_pca(X, k): compute col mean, center X, solve sorted eigh on cov(X_c), return mean and top-k cols W
def fit_pca(X, k):
    mean = np.mean(X, axis=0)
    X_c = X - mean
    n = X.shape[0]
    C = (X_c.T @ X_c) / (n - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    W = vecs[:, idx][:, :k]
    return mean, W

# 2. transform(X_new, mean, W): return (X_new - mean) @ W
def transform(X_new, mean, W):
    return (X_new - mean) @ W

# 3. Define training data X and new agent B = [[6, 0]]
X = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]], dtype=float)
B = np.array([[6, 0]], dtype=float)

# 4. Fit model: mean, W = fit_pca(X, k=1)
mean, W = fit_pca(X, k=1)

# 5. Transform and print scores for training X (A) and new agent B (B)
Z_B = transform(B, mean, W)
Z_A = transform(X, mean, W)

print("Edge Case B Score:\n", np.round(Z_B, 4))
print("Training A Scores:\n", np.round(Z_A, 4))