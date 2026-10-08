import numpy as np

def classify_matrix(A):
    
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return None, "not square"
    d = float(np.linalg.det(A))
    if np.isclose(d, 0.0, atol=1e-8):
        return d, "singular"
    return d, "invertible"

for name, mat in [
    ("R", [[1, 2, 3], [4, 5, 6]]),
    ("Z", [[0, 0], [0, 0]]),
    ("P", [[1, 2], [2, 4]]),
    ("Q", [[1, 2], [2, 4.0001]])
]:
    d, label = classify_matrix(mat)
    print(f"{name} -> det: {d}, label: {label}")