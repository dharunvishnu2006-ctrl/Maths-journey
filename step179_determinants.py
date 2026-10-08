import numpy as np

def det_2x2(M):
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    return a * d - b * c

A = np.array([[3, 1], [2, 4]])
print("By hand :", det_2x2(A))
print("NumPy   :", np.linalg.det(A))

S = np.array([[2, 4], [1, 2]])
print("det(S) by hand :", det_2x2(S))
print("det(S) NumPy   :", np.linalg.det(S))
print("Singular?      :", np.isclose(np.linalg.det(S), 0))

if np.isclose(np.linalg.det(S), 0.0):
    print("Cannot invert S: Matrix is singular (det == 0)")
else:
    print(np.linalg.inv(S))

for name, mat in [
    ("I", np.eye(3)),
    ("M", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
    ("N", np.array([[1, 2, 3], [4, 5, 6]])),
]:
    try:
        d = np.linalg.det(mat)
        print(f"{name} det: {d}, singular: {np.isclose(d, 0)}")
    except np.linalg.LinAlgError as e:
        print(f"{name} error: {e}")

def check_invertibility(X):
    n, p = X.shape

    if n < p:
        return 0.0, False

    G = X.T @ X

    d = float(np.linalg.det(G))

    is_singular = np.isclose(d, 0.0, atol=1e-8)

    safe = bool(not is_singular)
    return d, safe

X1 = np.array([
    [1, 2, 0],
    [2, 0, 1],
    [0, 1, 3],
    [3, 1, 1],
    [1, 3, 2]
], dtype=float)

d1, safe1 = check_invertibility(X1)
print(f"X1 -> det: {d1:.4f}, safe: {safe1}")

X2 = np.array([
    [1, 2, 3],
    [2, 0, 2],
    [0, 1, 1],
    [3, 1, 4],
    [1, 3, 4]
], dtype=float)

d2, safe2 = check_invertibility(X2)
print(f"X2 -> det: {d2:.4e}, safe: {safe2}")

X_edge = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float)

d3, safe3 = check_invertibility(X_edge)
print(f"X_edge (2x3) -> det: {d3:.4f}, safe: {safe3}")        