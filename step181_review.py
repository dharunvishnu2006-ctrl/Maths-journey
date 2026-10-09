import numpy as np

def pca_report(X):
    # 1. Edge case: if n < 2, return None, None
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) < 2:
        return None, None
    # 2. Centre data
    X_c = X - X.mean(axis=0)
    # 3. Covariance
    C = (X_c.T @ X_c) / (len(X) - 1)
    # 4. Eigendecompose & sort descending; clamp negatives
    vals = np.maximum(np.linalg.eigh(C)[0], 0.0)
    vals = np.sort(vals)[::-1]
    # 5. Check total variance
    if vals.sum() <= 1e-9:
        return None, None
    # 6. Ratios
    return vals, vals / vals.sum()

# 7. Test edge cases first: D, B, then A
for name, data in [("D", [[1, 2]]), ("B", [[4, 2], [4, 5], [4, 9]]), ("A", [[1, 3], [3, 1], [5, 7], [7, 5]])]:
    v, r = pca_report(data)
    print(f"{name}: vals={np.round(v, 4) if v is not None else None}, ratios={np.round(r, 4) if r is not None else None}")