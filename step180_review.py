import numpy as np

def top_direction(C):
  C = np.array(C, dtype=float)

  if not np.allclose(C, C.T):
    print("Matrix is not symmetric.")
    return None, None
  vals, vecs = np.linalg.eigh(C)
  top_lambda = vals[-1]
  top_vector = vecs[:, -1]

  return top_lambda, top_vector

N = [[1, 2], [0, 1]]
I = np.eye(3)
S = [[2, 1], [1, 2]]

print("Test 1 (N - Asymmetric):")
lam_n, vec_n = top_direction(N)
print(f"Result: lambda = {lam_n}, vector = {vec_n}\n")

print("Test 2 (I - Identity 3x3):")
lam_i, vec_i = top_direction(I)
print(f"Result: lambda = {lam_i}, vector = {vec_i}\n")

print("Test 3 (S - Symmetric):")
lam_s, vec_s = top_direction(S)
print(f"Result: lambda = {lam_s}, vector = {vec_s}")