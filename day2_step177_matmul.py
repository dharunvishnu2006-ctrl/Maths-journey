import numpy as np


def matmul(A, B):
    m, n = A.shape
    n2, p = B.shape

    if n != n2:
        raise ValueError(f"Inner dimensions don't match: {A.shape} @ {B.shape}")

    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


def check_matmul(name, A, B):
    mine = matmul(A, B)
    lib = A @ B
    print(f"--- {name} ---")
    print("Mine:\n", mine)
    print("Library:\n", lib)
    print("Shapes:", mine.shape, lib.shape)
    print("Match:", np.allclose(mine, lib))
    return mine, lib



C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])

E = np.array([[1, 0, 2], [0, 1, 3]])
F = np.array([[1, 2], [0, 1], [4, 0]])

cd_mine, cd_lib = check_matmul("C @ D", C, D)
check_matmul("E @ F", E, F)
check_matmul("F @ E", F, E)

dc_mine, dc_lib = check_matmul("D @ C", D, C)
print(f"\nIs C @ D == D @ C? {np.allclose(cd_mine, dc_mine)}")

print("\n--- Testing Custom Guard: matmul(E, E) ---")
try:
    matmul(E, E)
except ValueError as err:
    print("Custom error:", err)

print("\n--- Testing NumPy Guard: E @ E ---")
try:
    _ = E @ E
except ValueError as err:
    print("NumPy error:", err)


def validate_non_negative(name, M):
    if np.any(M < 0):
        raise ValueError(f"Matrix '{name}' contains negative values.")


O = np.array([
    [2, 0],
    [1, 4],
    [0, 3]
])

R = np.array([
    [0.2, 0.25, 0.0],
    [0.0, 0.0,  0.1]
])

P = np.array([
    [60,  55],
    [240, 260],
    [40,  45]
])

validate_non_negative("Orders (O)", O)
validate_non_negative("Recipe (R)", R)
validate_non_negative("Prices (P)", P)

if O.shape[1] != R.shape[0]:
    raise ValueError(f"Shape mismatch: O columns ({O.shape[1]}) != R rows ({R.shape[0]})")

if R.shape[1] != P.shape[0]:
    raise ValueError(f"Shape mismatch: R columns ({R.shape[1]}) != P rows ({P.shape[0]})")

print("\nTask 3 (comments 1-3) passed: all matrices defined, non-negative, and shapes align.")

ingredients = matmul(O, R)
cost_1 = matmul(ingredients, P)

dish_costs = matmul(R, P)
cost_2 = matmul(O, dish_costs)

associative_match = np.allclose(cost_1, cost_2)
numpy_match = np.allclose(cost_1, O @ R @ P)

print("Ingredients (O @ R, kg):\n", ingredients)
print("\nDish Costs (R @ P, ₹/dish):\n", dish_costs)
print("\nCost 1 ((O @ R) @ P):\n", cost_1)
print("\nCost 2 (O @ (R @ P)):\n", cost_2)
print("\n(O @ R) @ P == O @ (R @ P)?", associative_match)
print("Matches NumPy O @ R @ P?", numpy_match)


print("\n--- Cost Table (₹) ---")
print(f"{'Customer':<12} | {'Supplier 1':<12} | {'Supplier 2':<12}")
print("-" * 42)

for idx, row in enumerate(cost_1):
    c1, c2 = row[0], row[1]
    print(f"Customer {idx + 1:<3} | ₹{c1:<11.2f} | ₹{c2:<11.2f}")

print("\n--- Cheaper Supplier Recommendations ---")
for idx, row in enumerate(cost_1):
    c1, c2 = row[0], row[1]
    if np.isclose(c1, c2):
        print(f"Customer {idx + 1}: Either supplier (Both ₹{c1:.2f})")
    elif c1 < c2:
        print(f"Customer {idx + 1}: Supplier 1 (₹{c1:.2f} vs ₹{c2:.2f})")
    else:
        print(f"Customer {idx + 1}: Supplier 2 (₹{c2:.2f} vs ₹{c1:.2f})")