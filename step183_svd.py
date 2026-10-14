import numpy as np

A = np.array([[3, 0],
              [4, 5]])

U, S, Vt = np.linalg.svd(A)

print("U =\n", U)
print("S =", S)
print("Vt =\n", Vt)

Sigma = np.diag(S)
A_rebuilt = U @ Sigma @ Vt

print("Sigma =\n", Sigma)
print("A_rebuilt =\n", A_rebuilt)
print("Same as A?", np.allclose(A, A_rebuilt))
print("U.T @ U =\n", np.round(U.T @ U, 10))

k = 1
A_1 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

print("A_1 =\n", A_1)
print("Error =", np.linalg.norm(A - A_1))

B = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(B)

print(f"Shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
print(f"Singular values: {S}")
print("Valid S:", bool(np.all(S >= 0) and np.all(S[:-1] >= S[1:])))
print(f"Share σ₁: {(S[0]**2 / np.sum(S**2)) * 100:.2f}%")

B = np.array([[1, 2],
              [3, 4],
              [5, 6]])

U, S, Vt = np.linalg.svd(B)

Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)

B_rebuilt = U @ Sigma @ Vt

print("B_rebuilt:\n", B_rebuilt)

print("Rebuilt matches B:", np.allclose(B, B_rebuilt))