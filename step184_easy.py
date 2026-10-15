import numpy as np

# 1. Define coefficient matrix A and constants vector b
A = np.array([[ 1,  1,  1],
              [ 2, -1,  1],
              [ 1,  2, -1]], dtype=float)
b = np.array([6, 3, 2], dtype=float)

# 2. Solve the linear system A x = b using np.linalg.solve
x = np.linalg.solve(A, b)

# 3. Verify that A @ x equals b using np.allclose
is_verified = np.allclose(A @ x, b)

# 4. Compute determinant and condition number of A
det_A = np.linalg.det(A)
cond_A = np.linalg.cond(A)

# 5. Print results
print("x:           ", np.round(x, 4))
print("A @ x == b?  ", is_verified)
print("det(A):      ", np.round(det_A, 4))
print("cond(A):     ", np.round(cond_A, 4))