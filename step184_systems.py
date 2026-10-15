import numpy as np

A = np.array([[2, 1],
              [1, 2]], dtype=float)
b = np.array([110, 100], dtype=float)

x_inv = np.linalg.inv(A) @ b
x_solve = np.linalg.solve(A, b)

print("x via inverse:", x_inv)
print("x via solve:  ", x_solve)
print("A @ x == b?", np.allclose(A @ x_solve, b))

A3 = np.array([[2, 1],
               [1, 2],
               [1, 1]], dtype=float)
b3 = np.array([110, 100, 72], dtype=float)

x_ls, residual_ss, rank, sv = np.linalg.lstsq(A3, b3, rcond=None)

print("least squares x:", np.round(x_ls, 4))
print("predicted bills:", np.round(A3 @ x_ls, 3))
print("sum of squared misses:", np.round(residual_ss, 4))
print("rank:", rank)

A_good = np.array([[2, 1], [1, 2]], dtype=float)
A_bad  = np.array([[1, 1], [1, 1.001]], dtype=float)

print("cond(hotel bills):", round(np.linalg.cond(A_good), 2))
print("cond(near-twins): ", round(np.linalg.cond(A_bad), 1))