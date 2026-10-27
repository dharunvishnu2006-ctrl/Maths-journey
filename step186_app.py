import sympy as sp

# ==========================================
# Plan Comments:
# 1. Define symbolic variable w and loss expression L = (3*w - 6)**2
# 2. Compute exact symbolic derivative dL/dw using sp.diff()
# 3. Print the symbolic derivative formula
# 4. Evaluate and print L'(w) at w = 0, 1, 2, 3 using .subs()
# 5. Define Python function L_func(w) returning (3*w - 6)**2
# 6. Define numeric_derivative(f, x, h) from Step 185
# 7. Compute and print numeric slope at w = 1 with h = 0.001
# ==========================================

# 1. Define symbol and loss function
w = sp.Symbol('w')
L = (3 * w - 6) ** 2

# 2. Compute exact derivative
dL = sp.diff(L, w)

# 3 & 4. Print formula and evaluations
print("--- Part 1: Symbolic Derivative ---")
print(f"L'(w) formula: {dL}")
for val in [0, 1, 2, 3]:
    slope_val = dL.subs(w, val)
    print(f"L'({val}) = {slope_val}")

# 5. Define Python function for numeric evaluation
def L_func(w_val):
    return (3 * w_val - 6) ** 2

# 6. Numeric derivative helper
def numeric_derivative(f, x, h=1e-3):
    return (f(x + h) - f(x)) / h

# 7. Numeric slope at w = 1
num_slope = numeric_derivative(L_func, 1, h=0.001)

print("\n--- Part 2: Numeric Check at w = 1 ---")
print(f"Numeric derivative (h=0.001): {num_slope:.6f}")