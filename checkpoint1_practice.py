import numpy as np

# P1: Vectors
u = np.array([3, -1, 2])
v = np.array([1, 4, -2])
dot_uv = u @ v
if np.isclose(dot_uv, 0.0):
    angle_str = "= 90 deg (orthogonal)"
elif dot_uv < 0:
    angle_str = "> 90 deg (obtuse)"
else:
    angle_str = "< 90 deg (acute)"
print(f"P1 u.v = {dot_uv}, angle = {angle_str}")

# P2: Cosine Similarity
a = np.array([1, 2, 2])
b = np.array([2, 1, 2])
cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"P2 cos(theta) = {cos_theta:.4f}")

# P3: Matrix Multiplication Shapes & Values
A3 = np.array([[1, 0, -2], [3, 1, 4]])
B3 = np.array([[2, 1], [-1, 0], [1, 3]])
AB3 = A3 @ B3
BA3 = B3 @ A3
print(f"P3 A@B shape = {AB3.shape}, B@A shape = {BA3.shape}")
print(f"P3 A@B =\n{AB3}")

# P4: Order Matters
P = np.array([[1, 2], [0, 1]])
Q = np.array([[0, 1], [1, 0]])
PQ = P @ Q
QP = Q @ P
print(f"P4 P@Q =\n{PQ}")
print(f"P4 Q@P =\n{QP}")

# P5: Inverse & Determinant Check
A5 = np.array([[3, 1], [5, 2]], dtype=float)
det_A5 = np.linalg.det(A5)
inv_A5 = np.linalg.inv(A5)
check_I = A5 @ inv_A5
print(f"P5 det(A) = {det_A5:.1f}, det_matches_1 = {np.isclose(det_A5, 1.0)}")
print(f"P5 A_inv =\n{inv_A5}")
print(f"P5 A@A_inv == I: {np.allclose(check_I, np.eye(2))}")

# P6: Singularity & k
k = 3
C6 = np.array([[2, k], [6, 9]], dtype=float)
det_C6 = np.linalg.det(C6)
print(f"P6 k = {k}, is_singular = {np.isclose(det_C6, 0.0)}")

# P7: Eigenvalues with np.linalg.eig
A7 = np.array([[1, 2], [2, 1]], dtype=float)
eigvals7, eigvecs7 = np.linalg.eig(A7)
# Sort to inspect λ = 3 and λ = -1 clearly
sorted_idx7 = np.argsort(eigvals7)[::-1]
lams7 = eigvals7[sorted_idx7]
print(f"P7 eigvals = {lams7.round().astype(int).tolist()}")
for idx in range(len(lams7)):
    val = eigvals7[idx]
    vec = eigvecs7[:, idx]
    assert np.allclose(A7 @ vec, val * vec), "Av != lambda v failed"
print("P7 verification Av == lambda*v for all elements: True")

# P8: Eigenvalues in Code (eigh for symmetric matrix)
vals8, vecs8 = np.linalg.eigh(A7)
top_vec8 = vecs8[:, -1]
expected_mag = 1.0 / np.sqrt(2)
matches_top8 = np.allclose(np.abs(top_vec8), np.array([expected_mag, expected_mag]))
print(f"P8 eigh vals = {vals8}, top_vec = {top_vec8.round(4)}, matches ±[0.7071, 0.7071]: {matches_top8}")

# P9: PCA When Compression Fails
X9 = np.array([[1, 1], [1, 3], [3, 1], [3, 3]], dtype=float)
mu9 = np.mean(X9, axis=0)
Xc9 = X9 - mu9
cov9 = (Xc9.T @ Xc9) / (len(X9) - 1)
vals9, _ = np.linalg.eigh(cov9)
ratios9 = vals9[::-1] / np.sum(vals9)
k_needed = 2 if ratios9[0] < 0.95 else 1
print(f"P9 ratios = {ratios9}, k_95% = {k_needed}")

# P10: PCA Projection of New Agent
w1 = np.array([0.6, 0.8])
w2 = np.array([-0.8, 0.6])
norm_w1 = np.linalg.norm(w1)
norm_w2 = np.linalg.norm(w2)
dot_w1_w2 = w1 @ w2
mu_train = np.array([10, 50])
x_new = np.array([13, 54])
xc_new = x_new - mu_train
W = np.column_stack([w1, w2])
z = xc_new @ W
print(f"P10 ||w1|| = {norm_w1:.0f}, ||w2|| = {norm_w2:.0f}, w1.w2 = {dot_w1_w2:.0f}")
print(f"P10 xc = {xc_new.tolist()}, z = {z.tolist()}")