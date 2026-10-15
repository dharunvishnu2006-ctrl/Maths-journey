import numpy as np

# 1. Define units data and bill amounts (b)
units = np.array([100, 150, 200, 250, 300], dtype=float)
b = np.array([520, 740, 980, 1210, 1430], dtype=float)

# 2. Build design matrix A (shape 5x2) with bias column of 1s and units column
A = np.column_stack([np.ones_like(units), units])

# 3. Solve via Normal Equation using np.linalg.solve(A.T @ A, A.T @ b) without using inv
ATA = A.T @ A
ATb = A.T @ b
x_norm = np.linalg.solve(ATA, ATb)

# 4. Solve directly using np.linalg.lstsq(A, b, rcond=None)
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# 5. Verify both parameter solutions match using np.allclose
matches = np.allclose(x_norm, x_lstsq)

# 6. Compute and print sum of squared misses (SSE)
sse = np.sum((b - A @ x_norm) ** 2)

# 7. Compute and print cond(A) and cond(A.T @ A)
cond_A = np.linalg.cond(A)
cond_ATA = np.linalg.cond(ATA)

print(f"Normal Eq:  Fixed = ₹{x_norm[0]:.2f}, Rate = ₹{x_norm[1]:.4f}/unit")
print(f"lstsq:      Fixed = ₹{x_lstsq[0]:.2f}, Rate = ₹{x_lstsq[1]:.4f}/unit")
print(f"Matches?    {matches}")
print(f"SSE:        {sse:.4f}")
print(f"cond(A):    {cond_A:.4f}")
print(f"cond(ATA):  {cond_ATA:.4f}")