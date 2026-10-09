import numpy as np

B = np.array([[2, 1], [1, 2]])
vals, vecs = np.linalg.eig(B)

print("Eigenvalues :", vals)
print("Eigenvectors:")
print(vecs)

for i in range(len(vals)):
    v = vecs[:, i]
    lam = vals[i]
    print(f"pair {i}: B@v = {B @ v}, lam*v = {lam * v}, match = {np.allclose(B @ v, lam * v)}")

D = np.array([[5, 2], [2, 2]])
vals, vecs = np.linalg.eig(D)

print("Eigenvalues :", vals)
print("Eigenvectors:")
print(vecs)

for i in range(len(vals)):
    v, lam = vecs[:, i], vals[i]
    print(
        f"pair {i}: D@v = {D @ v}, lam*v = {lam * v}, match = {np.allclose(D @ v, lam * v)}"
    )    

G = np.array([[1, 4], [2, 3]])

vals, vecs = np.linalg.eig(G)

idx = np.argsort(vals)[::-1]
vals_sorted = vals[idx]
vecs_sorted = vecs[:, idx]

print("Sorted eigenvalues :", vals_sorted)
print("Sorted eigenvectors:")
print(vecs_sorted)

for i in range(len(vals_sorted)):
    v = vecs_sorted[:, i]
    lam = vals_sorted[i]
    match = np.allclose(G @ v, lam * v)
    print(f"pair {i} (lambda = {lam:.1f}): match = {match}")    

C = np.array([[4.0, 2.0, 0.6], [2.0, 3.0, 0.4], [0.6, 0.4, 1.0]])
feature_names = ["tasks per hour", "CPU load", "error rate"]

is_symmetric = np.allclose(C, C.T)
print(f"1. Is C symmetric? {is_symmetric}")

vals, vecs = np.linalg.eig(C)

idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]
print(f"2. Sorted eigenvalues: {vals}")
print(f"   Sum of eigenvalues: {vals.sum():.4f} (Trace = {np.trace(C):.4f})")

for i in range(len(vals)):
    v = vecs[:, i]
    lam = vals[i]
    match = np.allclose(C @ v, lam * v)
    print(f"3. Pair {i} (lambda = {lam:.4f}): Cv == lam*v is {match}")

ortho_checks = []
for i in range(len(vals)):
    for j in range(i + 1, len(vals)):
        dot_product = np.dot(vecs[:, i], vecs[:, j])
        is_ortho = np.isclose(dot_product, 0.0, atol=1e-7)
        ortho_checks.append(is_ortho)
        print(f"4. v_{i} . v_{j} = {dot_product:.2e} (orthogonal: {is_ortho})")

top_v = vecs[:, 0]
top_feature_idx = np.argmax(np.abs(top_v))
top_feature = feature_names[top_feature_idx]

print(f"5. Top eigenvector: {top_v}")
print(
    f"   Dominant feature: {top_feature} (weight = {abs(top_v[top_feature_idx]):.4f})"
)    