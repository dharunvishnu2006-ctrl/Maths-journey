import numpy as np

km = np.array([2, 4, 6, 8], dtype=float)
fare = np.array([40, 56, 72, 88], dtype=float)

A = np.column_stack([np.ones_like(km), km])
x, *_ = np.linalg.lstsq(A, fare, rcond=None)

base, rate = x[0], x[1]
print(f"base = {base:.2f}, rate = {rate:.2f}")
print("fare for 5 km =", base + rate * 5)

import numpy as np

A = np.array([[2, 1], [1, 2]], dtype=float)
b = np.array([110, 100], dtype=float)

x = np.linalg.solve(A, b)
print(x)