import sympy as sp

# 1. Define symbolic variable x
x = sp.Symbol('x')

# 2. Define numeric_derivative(f, x_val, h=0.001) using (f(x_val + h) - f(x_val)) / h
def numeric_derivative(f, x_val, h=0.001):
    return (f(x_val + h) - f(x_val)) / h

# 3. For each problem (a), (b), (c):
problems = [
    (
        "(a) Product: x^2 * (2x + 1)",
        x**2 * (2 * x + 1),
        lambda val: (val**2) * (2 * val + 1),
        1.0,
    ),
    (
        "(b) Quotient: (x + 1) / (x - 1)",
        (x + 1) / (x - 1),
        lambda val: (val + 1) / (val - 1),
        3.0,
    ),
    (
        "(c) Chain: (2x - 1)^4",
        (2 * x - 1) ** 4,
        lambda val: (2 * val - 1) ** 4,
        1.0,
    ),
]

for label, sym_expr, py_func, test_pt in problems:
    # b. Compute exact derivative formula with sp.diff()
    df_sym = sp.diff(sym_expr, x)

    # c. Evaluate exact value at test point using .subs()
    exact_val = float(df_sym.subs(x, test_pt))

    # d. Evaluate numeric derivative at test point with h = 0.001
    numeric_val = numeric_derivative(py_func, test_pt, h=0.001)

    # e. Compare absolute error: if abs(exact - numeric) < 0.05 print "OK", else "CHECK"
    diff = abs(exact_val - numeric_val)
    status = "OK" if diff < 0.05 else "CHECK"

    print(f"{label} at x = {test_pt}")
    print(f"  Formula : {df_sym}")
    print(f"  Exact   : {exact_val:.6f}")
    print(f"  Numeric : {numeric_val:.6f}")
    print(f"  Diff    : {diff:.6f} -> [{status}]\n")