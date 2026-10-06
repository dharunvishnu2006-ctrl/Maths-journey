import numpy as np

def inverse_2x2(A):
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"Expected shape (2, 2), got {A.shape}")

    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    det = a * d - b * c

    if abs(det) < 1e-9:
        raise ValueError(
            "Matrix is singular (determinant is zero) and cannot be inverted."
        )

    return (1.0 / det) * np.array([[d, -b], [-c, a]])


A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

print("A shape:", A.shape)
print("A.T shape:", A.T.shape)
print("A.T:\n", A.T)

B = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
print("(A @ B).T == B.T @ A.T?", np.allclose((A @ B).T, B.T @ A.T))

orders = np.array([[2, 1], [1, 3]], dtype=float)
bills = np.array([35, 55], dtype=float)

orders_inv = inverse_2x2(orders)
prices = orders_inv @ bills

print("\nOrders inverse:\n", orders_inv)
print("Recovered prices (tea, vada):", prices)
print("orders @ orders_inv ≈ I?", np.allclose(orders @ orders_inv, np.eye(2)))


# ===== Task 1 =====
print("\n--- Part 1: C ---")
C = np.array([[4, 7], [2, 6]], dtype=float)

C_inv_custom = inverse_2x2(C)
C_inv_numpy = np.linalg.inv(C)

I2 = np.eye(2)
print("C_inv (custom):\n", C_inv_custom)
print("C @ C_inv == I?", np.allclose(C @ C_inv_custom, I2))
print("C_inv @ C == I?", np.allclose(C_inv_custom @ C, I2))
print("Custom matches NumPy?", np.allclose(C_inv_custom, C_inv_numpy))

print("\n--- Part 2: S (Singular Matrix) ---")
S = np.array([[2, 4], [1, 2]], dtype=float)

try:
    inverse_2x2(S)
except (ValueError, np.linalg.LinAlgError) as err:
    print(f"inverse_2x2 caught: {type(err).__name__}: {err}")

try:
    np.linalg.inv(S)
except (ValueError, np.linalg.LinAlgError) as err:
    print(f"np.linalg.inv caught: {type(err).__name__}: {err}")


# ===== Task 2 =====
print("\n--- Task 2: Non-Square Matrix ---")
X = np.array([[2, 1], [1, 3], [3, 2]], dtype=float)

y = np.array([35, 55, 62], dtype=float)

try:
    np.linalg.inv(X)
except (ValueError, np.linalg.LinAlgError) as err:
    print(f"np.linalg.inv on non-square: {type(err).__name__}: {err}")

X_pinv = np.linalg.pinv(X)
print("\nX_pinv shape:", X_pinv.shape)
print("X_pinv:\n", X_pinv)

left_mult = X_pinv @ X
right_mult = X @ X_pinv

print("\nX_pinv @ X shape:", left_mult.shape)
print("X_pinv @ X:\n", left_mult)
print("X_pinv @ X == I(2)?", np.allclose(left_mult, np.eye(2)))

print("\nX @ X_pinv shape:", right_mult.shape)
print("X @ X_pinv:\n", right_mult)
print("X @ X_pinv == I(3)?", np.allclose(right_mult, np.eye(3)))

prices_task2 = X_pinv @ y
print("\nRecovered prices (tea, vada):", prices_task2)

lstsq_prices = np.linalg.lstsq(X, y, rcond=None)[0]
print("np.linalg.lstsq prices:", lstsq_prices)
print("Prices match lstsq?", np.allclose(prices_task2, lstsq_prices))

predicted_bills = X @ prices_task2
residuals = predicted_bills - y
print("\nPredicted bills:", predicted_bills)
print("Actual bills:", y)
print("Residuals (predicted - actual):", residuals)