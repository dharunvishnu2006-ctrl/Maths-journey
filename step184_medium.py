import numpy as np

# 1. Define coefficient matrix A (4x2) and vector b (4,)
A = np.array([[2, 1],
              [1, 3],
              [3, 2],
              [1, 1]], dtype=float)
b = np.array([50, 95, 85, 34], dtype=float)

# 2. Try solving with np.linalg.solve inside try/except block
try:
    x_solve = np.linalg.solve(A, b)
    print("solve x:", x_solve)
except np.linalg.LinAlgError as e:
    print("Step 1 Error:", type(e).__name__, "-", e)

# 3. Solve the overdetermined system via least squares
x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("Least squares x (p, n):", np.round(x_ls, 4))

# 4. Calculate misses (b - A @ x) and find worst-fitting bill
misses = b - (A @ x_ls)
print("Misses per bill (b - A @ x):", np.round(misses, 4))
worst_idx = np.argmax(np.abs(misses))
print(f"Worst-fitting bill: Bill {worst_idx + 1} with error of {misses[worst_idx]:.4f}")