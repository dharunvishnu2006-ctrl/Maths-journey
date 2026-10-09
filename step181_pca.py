import numpy as np

X = np.array([[1, 1], [2, 3], [3, 2], [4, 4]], dtype=float)

X_c = X - X.mean(axis=0)

C = np.cov(X_c.T)

vals, vecs = np.linalg.eigh(C)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

ratio = vals / vals.sum()

print("Mean :", X.mean(axis=0))
print("C    :\n", C)
print("vals :", vals)
print("PC1  :", vecs[:, 0])
print("ratio:", ratio)

X = np.array([
    [10, 50, 1],
    [12, 55, 2],
    [ 8, 45, 1],
    [14, 60, 3],
    [ 6, 40, 0]
], dtype=float)

n = X.shape[0]

mean = np.mean(X, axis=0)
X_c = X - mean

print("Calculated Mean:", mean)
print("X_c column means (approx 0):", np.mean(X_c, axis=0))

C_manual = (X_c.T @ X_c) / (n - 1)
C_builtin = np.cov(X_c.T)

matches = np.allclose(C_manual, C_builtin)
print("Manual matches np.cov:", matches)
print("C shape:", C_manual.shape)
print("\nCovariance Matrix C:\n", C_manual)

X = np.array([[10, 50, 1], [12, 55, 2], [8, 45, 1], [14, 60, 3], [6, 40, 0]], float)
X_c = X - X.mean(axis=0)
C = (X_c.T @ X_c) / (len(X) - 1)

vals, vecs = np.linalg.eigh(C)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

ratios = vals / vals.sum()

print("Vals:", np.round(vals, 4))
print("Ratios:", np.round(ratios, 4), "Sum:", ratios.sum())
print("Ortho (Identity):", np.allclose(vecs.T @ vecs, np.eye(3)))

def choose_k(X, threshold):
    if threshold <= 0.0:
        return 1, np.array([1.0]), True, True

    X_c = X - X.mean(axis=0)
    n, d = X.shape
    C = (X_c.T @ X_c) / (n - 1)

    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    ratios = vals / vals.sum()
    cum_ratios = np.cumsum(ratios)

    hit = np.where(cum_ratios >= threshold - 1e-9)[0]
    k = int(hit[0] + 1) if len(hit) > 0 else d

    sum_valid = np.isclose(ratios.sum(), 1.0)
    ortho_valid = np.allclose(vecs.T @ vecs, np.eye(d))

    return k, cum_ratios, sum_valid, ortho_valid

X = np.array([
    [20, 45, 3, 120],
    [25, 55, 4, 110],
    [18, 40, 2, 130],
    [30, 62, 6, 100],
    [22, 50, 5, 125],
    [28, 58, 3, 105]
], float)

for t in [0.0, 0.95, 0.99, 1.0]:
    k, cum, s_ok, o_ok = choose_k(X, t)
    print(f"t={t:<4} -> k={k} | cum={np.round(cum, 4)} | sum=1: {s_ok} | ortho: {o_ok}")