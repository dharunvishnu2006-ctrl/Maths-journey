import numpy as np

# 1. Define A = np.arange(1, 16).reshape(5, 3) * 1.0
A = np.arange(1, 16).reshape(5, 3) * 1.0

# 2. Decompose: U, S, Vt = np.linalg.svd(A)
U, S, Vt = np.linalg.svd(A)

print(f"Singular values: {S}\n")

# 3. For k in range(1, len(S) + 1):
for k in range(1, len(S) + 1):
    # a. Rebuild: A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # b. Measured error: err_meas = np.linalg.norm(A - A_k)
    err_meas = np.linalg.norm(A - A_k)

    # c. Theoretical error: err_theo = np.sqrt(np.sum(S[k:]**2))
    err_theo = np.sqrt(np.sum(S[k:]**2))

    # d. Print k, err_meas, err_theo, and np.isclose match
    matches = np.isclose(err_meas, err_theo)
    print(f"k = {k}: measured = {err_meas:.6e}, theory = {err_theo:.6e}, match = {matches}")